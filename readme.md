# Fusion Ops V8

**Deterministic Ops Console for Human-Gated Autonomy**

Fusion Ops V8 is a **replayable, tamper-evident, explainable operations console** designed for autonomy, sensor fusion, and high-risk decision environments.

It combines:
- Deterministic simulation
- Journaled state transitions
- Explainable trust + clarity scoring
- Threat doctrine enforcement
- Human-in-the-loop autonomy gating
- Tactical map visualization

This is not a demo toy — every tick, decision, and operator action is **auditable, replayable, and hash-verifiable**.

---

## Core Guarantees

- **Determinism**  
  Identical journal + config → identical state hash.

- **Tamper Evidence**  
  Hash-chained journal with optional HMAC signing.

- **Explainability**  
  Trust is decomposed into freshness, schema, dropout, drift, and operator penalties.

- **Human Authority**  
  Autonomy actions are gated by policy, integrity score, threat doctrine, and operator load.

- **Replay & Forensics**  
  Any moment can be reconstructed exactly via the journal.

---

## System Architecture (High Level)

Synthetic Feeds
↓
Schema Validation → Quarantine Lane
↓
Clarity Engine + Trust Fabric
↓
Threat Doctrine Evaluation
↓
Recommendations (Policy + Doctrine Gated)
↓
Human Decisions
↓
Autonomy Actions (or Blocked)

All state transitions are journaled.

---

## Features

- Multi-source telemetry (ship, air, incident, weather)
- Schema validation with reject/quarantine lanes
- Clarity metrics (volatility, instability, novelty)
- Trust fabric with blame attribution
- Threat doctrine (GREEN → BLACK)
- Human-gated recommendations
- Counterfactual visualization
- Deterministic replay capsules
- Frontend-ready UI state export

---

## Installation

### 1. Clone

```bash
git clone <repo-url>
cd fusion-ops-v8
2. Create Virtual Environment (Recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
3. Install Requirements
pip install -r requirements.txt
________________________________________
requirements.txt
Your requirements.txt should include:
streamlit>=1.32
pydeck>=0.8
numpy>=1.24
pandas>=2.0
Optional (only if using Mapbox styles):
mapbox
________________________________________
Running the App
streamlit run app.py
The console will open in your browser.
________________________________________
Environment Variables (Optional)
Journal HMAC Signing (Recommended for audits)
export FUSIONOPS_JOURNAL_SECRET="your-secret-key"
If set, every journal entry is HMAC-signed and verified on replay.
Mapbox (Optional)
export MAPBOX_API_KEY="your-mapbox-key"
If unset, the app falls back to Carto Dark Matter (no key required).
________________________________________
Replay & Audit
•	Replay slider reconstructs state at any tick
•	Consistency check verifies deterministic rebuilds
•	Replay capsule export includes:
o	Config snapshot
o	Full journal
o	Final state hash
•	UI contract export produces a frontend-ready JSON payload
________________________________________
Safety Model
Autonomy is constrained by:
•	Global integrity score
•	Threat doctrine level
•	Operator load
•	Explicit policy thresholds
BLACK or RED threat levels block autonomous side-effects by design.
________________________________________
Intended Use
Fusion Ops V8 is designed for:
•	Autonomy supervision
•	Sensor fusion validation
•	Safety-critical ops tooling
•	Audit-first autonomy research
•	Human-centered control systems
It is explicitly not designed for unsupervised or weaponized autonomy.
________________________________________
License
MIT
________________________________________
