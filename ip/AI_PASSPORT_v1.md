This document contains non-public strategic information.
Distribution is restricted.


SECTION: INTELLECTUAL PROPERTY & COMPLIANCE

Project: Slovak Steed
Owner: Maksym Skachkov
Date: 22 March 2026
Status: Active

--------------------------------------------------

1. IP STRATEGY OVERVIEW

Slovak Steed follows a hybrid intellectual property model:

- Public Layer (Open Source — Apache 2.0)
- Protected Layer (Trade Secret + Patent-Pending)

This approach ensures:
- transparency for collaboration
- protection of core technological advantage
- readiness for commercial deployment

--------------------------------------------------

2. LAYER SEPARATION MODEL

PUBLIC LAYER (Apache 2.0)
Includes:
- simulation environments
- UI dashboards and visualization
- testing infrastructure
- CI/CD pipelines
- non-critical agent prompts

Purpose:
- ecosystem building
- validation
- collaboration

----------------------------------------

PROTECTED LAYER (Confidential)

Includes:
- reward_function.py
- reinforcement learning training pipeline
- domain randomization parameters
- rider-weight steering control logic (IP-OPP-001)
- adaptive locomotion system (IP-OPP-002)
- trained model weights
- internal knowledge base (master_kb.json)
- IP strategy documents

Protection type:
- trade secret
- patent-pending technology

Access:
- restricted
- not published
- controlled under NDA (if shared)

--------------------------------------------------

3. CLEAN ROOM DEVELOPMENT

The project enforces a strict clean-room development policy:

- External open-source implementations are not used as source code references
- Internal algorithms are developed independently
- Development is based on scientific literature and system requirements

Status:
- reward_function.py v2.0 confirmed as clean-room implementation

Purpose:
- ensure originality
- preserve patentability
- eliminate derivative work risks

--------------------------------------------------

4. OPEN-SOURCE COMPLIANCE

Allowed licenses:
- MIT
- BSD-2 / BSD-3
- Apache 2.0

Restricted:
- GPL / AGPL (reference only)

Rules:
- no copying of core logic
- all external code must be attributed
- dependency locking required
- license verification mandatory before use

--------------------------------------------------

5. DEPENDENCY CONTROL

All external dependencies are tracked in:

DEPENDENCY_LOCK.json

Each entry includes:
- repository
- license
- commit hash
- usage type
- approval status

Unverified or risky dependencies are blocked until review.

--------------------------------------------------

6. PATENT STRATEGY

Target jurisdictions:
- European Union (EPO / ÚPV SR)
- United States (USPTO)
- International (WIPO PCT)

Focus areas:
- human-machine coupling (body-weight control)
- adaptive locomotion systems
- control architecture
- training methodology (partial)

Key principle:
- patent what can be reverse-engineered
- keep internal what cannot

--------------------------------------------------

7. FREEDOM TO OPERATE (FTO)

The project acknowledges that:

- open-source ≠ patent-free
- control systems may be patented independently of code

Measures:
- prior art analysis
- FTO checks before integration
- avoidance of direct algorithm replication
- independent implementation

High-risk systems (e.g., vendor SDKs) are restricted pending legal review.

--------------------------------------------------

8. TRADE SECRET PROTECTION

Protected elements:
- control algorithms
- reward function design
- training strategies
- model weights

Security measures:
- private repositories
- access control
- encrypted storage
- no public exposure

--------------------------------------------------

9. INCIDENT RESPONSE

If protocol violation occurs:

1. Immediate stop of affected process
2. No commit or distribution
3. Report to project owner
4. Legal assessment
5. Incident logged

--------------------------------------------------

10. COMPLIANCE STATUS

Current status:
- Protocol approved and active
- Clean-room policy enforced
- Dependency tracking active
- IP structure defined
- Patent preparation in progress

Risk level:
CONTROLLED

--------------------------------------------------

DECLARATION

This document confirms that Slovak Steed is being developed with
a structured intellectual property strategy, ensuring:

- originality of core technology
- compliance with open-source licenses
- readiness for patent protection
- protection of commercial value

--------------------------------------------------
