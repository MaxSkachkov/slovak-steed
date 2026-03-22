# slovak-steed
# 🐴 Slovak Steed

> *Štyri nohy. Žiadne hranice.* — Four legs. No limits.

**Open-source quadruped transport platform.**  
Four mechanical legs instead of wheels.  
Maximum off-road capability with a human rider.

**First model: WRIGHT** *(named after the Wright Brothers)*

---

## Status

🟡 **Phase 1** — Digital prototype in MuJoCo simulation  
🔒 **Core AI algorithm** — Patent pending (filed April 2026)  
📅 **Started** — 21 March 2026

## What is Slovak Steed?

Not a robot animal. An industrial transport machine.  
Think KTM enduro meets Boston Dynamics — built in Slovakia.

- **Rider capacity:** 80 kg human rider  
- **System mass:** 200 kg total  
- **Power:** LiFePO4 electric (Phase 1), hydraulic (Phase 2)  
- **Interface:** 48V, CAN-bus, 4-point modular mount

## Architecture

```
Simulation:  MuJoCo 3.x + slovak_horse_v1.xml
Training:    Reinforcement Learning (PPO/SAC)
Monitoring:  Guardian v2 (IP protection)
MLflow:      Experiment tracking (Postgres + MinIO)
```

## Repository Structure

```
slovak-steed/
├── simulation/          # MuJoCo model
├── tests/               # CI test suite
├── scripts/             # Vast.ai GPU training
├── .github/workflows/   # CI/CD validation
├── legal/               # IP compliance docs
├── agents/              # AI team system prompts
├── docker-compose.yml   # MLflow stack
└── Dockerfile.guardian  # IP monitor service
```

## IP Notice

The core locomotion algorithm, body-weight steering system,  
and rider-coupling reward function are protected intellectual property.

See [`legal/IP_Protocol_v1.2.json`](legal/IP_Protocol_v1.2.json)

**Public layer:** Apache-2.0  
**Core algorithm:** BUSL-1.1 (proprietary)

## Quick Start

```bash
# Clone and validate simulation
git clone https://github.com/YOUR_ORG/slovak-steed
cd slovak-steed
pip install mujoco>=3.0.0 pytest numpy
python -c "import mujoco; mujoco.MjModel.from_xml_path('simulation/slovak_horse_v1.xml'); print('OK')"

# Start MLflow stack
cp .env.example .env
nano .env  # fill in your passwords
docker compose up -d
# MLflow UI: http://localhost:5000
```

## Contributing

Contributors welcome. Read [`legal/IP_Protocol_v1.2.json`](legal/IP_Protocol_v1.2.json)  
before contributing. Clean room rule applies.

---

*Built in Slovakia. Built with AI.*  
*Copyright (c) 2026 Maksym Skachkov*
