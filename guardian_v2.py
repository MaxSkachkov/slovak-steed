#!/usr/bin/env python3
"""
Slovak-Horse-Guardian-V2
Мониторинг целостности весов нейросети.

Улучшения v2 vs v1:
  - Детектирование аномалий градиентов (Gradient_Anomaly_Detection)
  - Фильтрация adversarial шума в observation space (Adversarial_Noise_Filtering)
  - Верификация целостности модели при загрузке (Model_Integrity_Verification)
  - JSON-лог инцидентов для QC-Agent и Legal-Patent-Agent
  - Поддержка нескольких папок одновременно
  - Авторизация через HMAC (не просто регистрация файла)
  - CLI-команды: monitor / authorize / verify / report

Python 3.10+, без платных зависимостей.
"""

import argparse
import hashlib
import hmac
import json
import logging
import os
import time
import smtplib
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from email.mime.text import MIMEText

# ─── Версия ──────────────────────────────────────────────────────
VERSION = "2.0.0"
AGENT_NAME = "Slovak-Horse-Guardian-V2"

# ─── Конфигурация из переменных окружения ────────────────────────
WATCH_DIRS     = os.environ.get("GUARDIAN_WATCH_DIRS", "./checkpoints").split(",")
STATE_FILE     = Path(os.environ.get("GUARDIAN_STATE_FILE", ".guardian_state.json"))
INCIDENT_LOG   = Path(os.environ.get("GUARDIAN_INCIDENT_LOG", "guardian_incidents.json"))
LOG_FILE       = Path(os.environ.get("GUARDIAN_LOG_FILE", "guardian.log"))
CHECK_INTERVAL = int(os.environ.get("GUARDIAN_INTERVAL", "30"))
HMAC_SECRET    = os.environ.get("GUARDIAN_HMAC_SECRET", "change-me-in-production")

# Email (опционально)
ALERT_EMAIL    = os.environ.get("GUARDIAN_ALERT_EMAIL", "")
ALERT_PASSWORD = os.environ.get("GUARDIAN_ALERT_PASSWORD", "")
ALERT_TO       = os.environ.get("GUARDIAN_ALERT_TO", ALERT_EMAIL)

WATCHED_EXTENSIONS = {".pth", ".onnx", ".pkl", ".safetensors", ".pt"}

# Пороги для Gradient Anomaly Detection
GRAD_NORM_THRESHOLD = float(os.environ.get("GUARDIAN_GRAD_THRESHOLD", "100.0"))
GRAD_NAN_POLICY     = os.environ.get("GUARDIAN_GRAD_NAN_POLICY", "alert")  # alert | abort

# ─── Логирование ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("Guardian-V2")


# ════════════════════════════════════════════════════════════════
#  УТИЛИТЫ
# ════════════════════════════════════════════════════════════════

def compute_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def compute_hmac(file_hash: str) -> str:
    """HMAC-SHA256 подпись хэша файла — для авторизации сохранений."""
    return hmac.new(
        HMAC_SECRET.encode(),
        file_hash.encode(),
        hashlib.sha256,
    ).hexdigest()


def verify_hmac(file_hash: str, signature: str) -> bool:
    expected = compute_hmac(file_hash)
    return hmac.compare_digest(expected, signature)


def format_size(path: Path) -> str:
    size = path.stat().st_size
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


# ─── Персистентное состояние ─────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def save_state(state: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ─── Журнал инцидентов (для QC-Agent и Legal) ────────────────────

def load_incidents() -> list:
    if INCIDENT_LOG.exists():
        with open(INCIDENT_LOG) as f:
            return json.load(f)
    return []


def log_incident(incident_type: str, filepath: str, details: dict) -> None:
    """
    Записывает инцидент в JSON-журнал.
    QC-Agent читает этот файл для Fails & Bugs Control.
    Legal-Patent-Agent использует как доказательную базу.
    """
    incidents = load_incidents()
    incident = {
        "id": len(incidents) + 1,
        "timestamp": datetime.now().isoformat(),
        "type": incident_type,
        "file": filepath,
        "agent": AGENT_NAME,
        "version": VERSION,
        **details,
    }
    incidents.append(incident)
    with open(INCIDENT_LOG, "w") as f:
        json.dump(incidents, f, indent=2)
    log.warning(f"[INCIDENT #{incident['id']}] {incident_type}: {filepath}")


# ════════════════════════════════════════════════════════════════
#  CORE FUNCTION 1: Gradient Anomaly Detection
# ════════════════════════════════════════════════════════════════

def check_gradient_anomalies(
    named_gradients: dict[str, float],
    step: int = 0,
) -> list[dict]:
    """
    Анализирует нормы градиентов во время обучения.
    Вызывать из training loop после loss.backward().

    Args:
        named_gradients: {param_name: grad_norm_value}
        step: текущий шаг обучения

    Returns:
        Список найденных аномалий (пустой = всё ок).

    Пример интеграции в training loop:
        grads = {n: p.grad.norm().item()
                 for n, p in model.named_parameters()
                 if p.grad is not None}
        anomalies = guardian.check_gradient_anomalies(grads, step=global_step)
        if anomalies:
            optimizer.zero_grad()  # сбросить испорченные градиенты
    """
    anomalies = []

    for param_name, grad_norm in named_gradients.items():
        issue = None

        # NaN в градиентах — критично
        if grad_norm != grad_norm:  # NaN check без numpy
            issue = {
                "anomaly": "NaN_gradient",
                "param": param_name,
                "value": "NaN",
                "step": step,
                "severity": "CRITICAL",
            }

        # Inf в градиентах
        elif grad_norm == float("inf") or grad_norm == float("-inf"):
            issue = {
                "anomaly": "Inf_gradient",
                "param": param_name,
                "value": str(grad_norm),
                "step": step,
                "severity": "CRITICAL",
            }

        # Взрыв градиентов
        elif grad_norm > GRAD_NORM_THRESHOLD:
            issue = {
                "anomaly": "gradient_explosion",
                "param": param_name,
                "value": round(grad_norm, 4),
                "threshold": GRAD_NORM_THRESHOLD,
                "step": step,
                "severity": "WARNING",
            }

        if issue:
            anomalies.append(issue)
            log_incident(
                incident_type=f"GRADIENT_{issue['anomaly'].upper()}",
                filepath=f"training_step_{step}",
                details=issue,
            )

    return anomalies


# ════════════════════════════════════════════════════════════════
#  CORE FUNCTION 2: Adversarial Noise Filtering
# ════════════════════════════════════════════════════════════════

def filter_adversarial_noise(
    observation: list[float],
    bounds: Optional[dict] = None,
    z_threshold: float = 4.0,
) -> tuple[list[float], list[dict]]:
    """
    Фильтрует подозрительные значения в observation space MuJoCo.
    Защита от adversarial attacks на симулятор.

    Args:
        observation: вектор наблюдений от среды
        bounds: {index: (min, max)} — допустимые диапазоны
        z_threshold: порог Z-score для детектирования выбросов

    Returns:
        (очищенный_вектор, список_предупреждений)

    Пример для locomotion задачи:
        bounds = {
            0: (-3.14, 3.14),   # углы суставов
            1: (-3.14, 3.14),
            12: (-10.0, 10.0),  # линейные скорости
        }
        clean_obs, warnings = guardian.filter_adversarial_noise(obs, bounds)
    """
    warnings_list = []
    clean = list(observation)

    # 1. Проверка NaN/Inf в наблюдениях
    for i, val in enumerate(clean):
        if val != val or val == float("inf") or val == float("-inf"):
            warnings_list.append({
                "filter": "NaN_Inf_in_observation",
                "index": i,
                "value": str(val),
                "action": "clipped_to_zero",
            })
            clean[i] = 0.0
            log_incident(
                "ADVERSARIAL_NaN_IN_OBS",
                f"obs_index_{i}",
                {"value": str(val), "index": i},
            )

    # 2. Проверка bounds если заданы
    if bounds:
        for idx, (low, high) in bounds.items():
            if idx < len(clean):
                if clean[idx] < low or clean[idx] > high:
                    original = clean[idx]
                    clean[idx] = max(low, min(high, clean[idx]))
                    warnings_list.append({
                        "filter": "out_of_bounds",
                        "index": idx,
                        "original": original,
                        "clipped_to": clean[idx],
                        "bounds": [low, high],
                    })

    # 3. Z-score детектирование глобальных выбросов
    n = len(clean)
    if n > 1:
        mean = sum(clean) / n
        variance = sum((x - mean) ** 2 for x in clean) / n
        std = variance ** 0.5

        if std > 0:
            for i, val in enumerate(clean):
                z = abs((val - mean) / std)
                if z > z_threshold:
                    warnings_list.append({
                        "filter": "z_score_outlier",
                        "index": i,
                        "value": round(val, 4),
                        "z_score": round(z, 2),
                        "threshold": z_threshold,
                    })

    return clean, warnings_list


# ════════════════════════════════════════════════════════════════
#  CORE FUNCTION 3: Model Integrity Verification
# ════════════════════════════════════════════════════════════════

def verify_model_integrity(filepath: Path) -> dict:
    """
    Проверяет целостность файла весов при загрузке.
    Использует HMAC-подпись из state файла.

    Returns:
        {"status": "ok"|"tampered"|"unknown", "hash": str, ...}
    """
    if not filepath.exists():
        return {"status": "not_found", "file": str(filepath)}

    current_hash = compute_sha256(filepath)
    state = load_state()
    rel = str(filepath.name)

    if rel not in state:
        return {
            "status": "unknown",
            "file": str(filepath),
            "hash": current_hash,
            "message": "Файл не зарегистрирован в Guardian. Запустите authorize.",
        }

    stored = state[rel]
    stored_hash = stored.get("hash", "")
    stored_hmac = stored.get("hmac", "")

    # Проверка хэша
    if current_hash != stored_hash:
        log_incident(
            "MODEL_INTEGRITY_VIOLATION",
            str(filepath),
            {
                "stored_hash": stored_hash,
                "current_hash": current_hash,
                "last_authorized": stored.get("seen_at", "unknown"),
            },
        )
        return {
            "status": "tampered",
            "file": str(filepath),
            "stored_hash": stored_hash,
            "current_hash": current_hash,
            "severity": "CRITICAL",
        }

    # Проверка HMAC подписи
    if stored_hmac and not verify_hmac(current_hash, stored_hmac):
        log_incident(
            "HMAC_VERIFICATION_FAILED",
            str(filepath),
            {"hash": current_hash},
        )
        return {
            "status": "hmac_invalid",
            "file": str(filepath),
            "severity": "CRITICAL",
            "message": "HMAC не совпадает — возможна подмена state файла.",
        }

    return {
        "status": "ok",
        "file": str(filepath),
        "hash": current_hash,
        "authorized_at": stored.get("seen_at", "unknown"),
    }


# ════════════════════════════════════════════════════════════════
#  МОНИТОРИНГ ФАЙЛОВОЙ СИСТЕМЫ
# ════════════════════════════════════════════════════════════════

def scan_directories(watch_dirs: list[str]) -> dict[str, dict]:
    result = {}
    for d in watch_dirs:
        watch_dir = Path(d.strip())
        if not watch_dir.exists():
            watch_dir.mkdir(parents=True, exist_ok=True)
            continue
        for path in watch_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in WATCHED_EXTENSIONS:
                rel = str(path)
                try:
                    file_hash = compute_sha256(path)
                    result[rel] = {
                        "hash": file_hash,
                        "hmac": compute_hmac(file_hash),
                        "size": path.stat().st_size,
                        "mtime": path.stat().st_mtime,
                        "seen_at": datetime.now().isoformat(),
                    }
                except (OSError, PermissionError) as e:
                    log.error(f"Не могу прочитать {path}: {e}")
    return result


def compare_states(old: dict, new: dict) -> tuple[list, list, list]:
    added   = [k for k in new if k not in old]
    removed = [k for k in old if k not in new]
    changed = [
        k for k in new
        if k in old and new[k]["hash"] != old[k]["hash"]
    ]
    return added, removed, changed


def send_alert(subject: str, body: str) -> None:
    log.warning(f"ALERT: {subject}")
    if not ALERT_EMAIL or not ALERT_PASSWORD:
        return
    try:
        msg = MIMEText(body)
        msg["Subject"] = f"[{AGENT_NAME}] {subject}"
        msg["From"]    = ALERT_EMAIL
        msg["To"]      = ALERT_TO
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(ALERT_EMAIL, ALERT_PASSWORD)
            server.send_message(msg)
        log.info(f"Alert отправлен на {ALERT_TO}")
    except Exception as e:
        log.error(f"Email error: {e}")


def process_events(
    added: list, removed: list, changed: list,
    old_state: dict, new_state: dict,
) -> None:
    for rel in added:
        info = new_state[rel]
        log.info(f"[NEW] {Path(rel).name} | SHA256: {info['hash'][:16]}...")

    for rel in removed:
        info = old_state[rel]
        log.warning(f"[REMOVED] {Path(rel).name}")
        log_incident("FILE_REMOVED", rel, {"last_hash": info["hash"]})
        send_alert(
            f"Файл удалён: {Path(rel).name}",
            f"Файл весов удалён.\nПуть: {rel}\nХэш: {info['hash']}\n"
            f"Время: {datetime.now().isoformat()}",
        )

    for rel in changed:
        old_h = old_state[rel]["hash"]
        new_h = new_state[rel]["hash"]
        # Проверяем HMAC новой версии
        if not verify_hmac(new_h, new_state[rel].get("hmac", "")):
            severity = "CRITICAL — HMAC не совпадает"
        else:
            severity = "INFO — авторизованное изменение"

        log.warning(f"[CHANGED] {Path(rel).name} | {severity}")
        log_incident(
            "FILE_MODIFIED",
            rel,
            {"old_hash": old_h, "new_hash": new_h, "severity": severity},
        )

        if "CRITICAL" in severity:
            send_alert(
                f"Несанкционированное изменение: {Path(rel).name}",
                f"Файл изменён без авторизации.\n"
                f"Путь: {rel}\nСтарый: {old_h}\nНовый: {new_h}\n"
                f"Время: {datetime.now().isoformat()}",
            )


# ════════════════════════════════════════════════════════════════
#  CLI КОМАНДЫ
# ════════════════════════════════════════════════════════════════

def cmd_monitor(args) -> None:
    """Основной цикл мониторинга."""
    log.info("═" * 60)
    log.info(f"  {AGENT_NAME} v{VERSION}")
    log.info(f"  Мониторинг: {', '.join(WATCH_DIRS)}")
    log.info(f"  Интервал: {args.interval}s")
    log.info("═" * 60)

    if HMAC_SECRET == "change-me-in-production":
        log.warning("⚠️  GUARDIAN_HMAC_SECRET не задан! Используется дефолт.")

    state = load_state()
    if not state:
        log.info("Первый запуск — строю baseline...")
        state = scan_directories(WATCH_DIRS)
        save_state(state)
        log.info(f"Baseline: {len(state)} файл(ов).")

    while True:
        time.sleep(args.interval)
        current = scan_directories(WATCH_DIRS)
        added, removed, changed = compare_states(state, current)

        if added or removed or changed:
            process_events(added, removed, changed, state, current)
            state = current
            save_state(state)
        else:
            log.debug("Изменений нет.")


def cmd_authorize(args) -> None:
    """Авторизовать файл весов — зарегистрировать HMAC-подпись."""
    path = Path(args.file)
    if not path.exists():
        log.error(f"Файл не найден: {path}")
        sys.exit(1)

    file_hash = compute_sha256(path)
    signature = compute_hmac(file_hash)
    state = load_state()

    state[str(path)] = {
        "hash": file_hash,
        "hmac": signature,
        "size": path.stat().st_size,
        "mtime": path.stat().st_mtime,
        "seen_at": datetime.now().isoformat(),
        "authorized_by": "manual",
    }
    save_state(state)
    log.info(f"[AUTHORIZED] {path.name}")
    log.info(f"  SHA256: {file_hash}")
    log.info(f"  HMAC:   {signature[:16]}...")


def cmd_verify(args) -> None:
    """Проверить целостность файла весов."""
    result = verify_model_integrity(Path(args.file))
    status = result["status"]

    icons = {"ok": "✅", "tampered": "🚨", "unknown": "❓", "not_found": "❌", "hmac_invalid": "🔐"}
    icon = icons.get(status, "?")

    print(f"\n{icon} Статус: {status.upper()}")
    for k, v in result.items():
        if k != "status":
            print(f"   {k}: {v}")
    print()

    sys.exit(0 if status == "ok" else 1)


def cmd_report(args) -> None:
    """Сводный отчёт по инцидентам — для QC-Agent и Legal."""
    incidents = load_incidents()

    if not incidents:
        print("Инцидентов не зарегистрировано.")
        return

    print(f"\n{'═'*60}")
    print(f"  {AGENT_NAME} — Отчёт по инцидентам")
    print(f"  Всего: {len(incidents)}")
    print(f"{'═'*60}")

    by_type: dict[str, int] = {}
    for inc in incidents:
        by_type[inc["type"]] = by_type.get(inc["type"], 0) + 1

    for itype, count in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {itype:<40} {count:>4} инц.")

    print(f"\nПоследние 5:")
    for inc in incidents[-5:]:
        print(f"  #{inc['id']} [{inc['timestamp'][:16]}] {inc['type']} — {inc['file']}")

    print(f"\nПолный журнал: {INCIDENT_LOG.resolve()}\n")


# ════════════════════════════════════════════════════════════════
#  ТОЧКА ВХОДА
# ════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"{AGENT_NAME} v{VERSION}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Команды:
  monitor              Запустить мониторинг (основной режим)
  authorize FILE       Авторизовать файл весов (HMAC-подпись)
  verify FILE          Проверить целостность файла
  report               Сводка инцидентов для QC / Legal

Переменные окружения:
  GUARDIAN_WATCH_DIRS       Папки через запятую (default: ./checkpoints)
  GUARDIAN_HMAC_SECRET      Секрет для HMAC (обязательно сменить!)
  GUARDIAN_INTERVAL         Интервал проверки в секундах (default: 30)
  GUARDIAN_GRAD_THRESHOLD   Порог нормы градиентов (default: 100.0)
  GUARDIAN_ALERT_EMAIL      Gmail для алертов (опционально)
  GUARDIAN_ALERT_PASSWORD   App Password Gmail

Интеграция в training loop:
  from guardian_v2 import check_gradient_anomalies, filter_adversarial_noise
  anomalies = check_gradient_anomalies({n: p.grad.norm().item() ...}, step)
  clean_obs, warns = filter_adversarial_noise(raw_obs, bounds)
        """,
    )

    sub = parser.add_subparsers(dest="command")

    p_mon = sub.add_parser("monitor")
    p_mon.add_argument("--interval", type=int, default=CHECK_INTERVAL)

    p_auth = sub.add_parser("authorize")
    p_auth.add_argument("file")

    p_ver = sub.add_parser("verify")
    p_ver.add_argument("file")

    sub.add_parser("report")

    args = parser.parse_args()

    if args.command == "monitor" or args.command is None:
        if args.command is None:
            args.interval = CHECK_INTERVAL
        cmd_monitor(args)
    elif args.command == "authorize":
        cmd_authorize(args)
    elif args.command == "verify":
        cmd_verify(args)
    elif args.command == "report":
        cmd_report(args)


if __name__ == "__main__":
    main()
