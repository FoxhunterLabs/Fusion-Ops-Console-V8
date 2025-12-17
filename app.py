# app.py
# ================================================================
# Fusion Ops V8 — Ops Console (Unified)
# Deterministic + Replayable + Governed + Tamper-Evident
# Tactical Map Surface + Threat Doctrine + Explainability + Human Gate
# ================================================================

from __future__ import annotations

import os
import json
import math
import time
import hmac
import hashlib
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Iterable, Set, Literal

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import streamlit.components.v1 as components

# ================================================================
# Page Config
# ================================================================

st.set_page_config(
    page_title="Fusion Ops V8",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ================================================================
# OPS UI: Dark doctrine, high contrast, ominous
# ================================================================

OPS_CSS = """
<style>
:root{
  --bg0:#05060a;
  --bg1:#070b10;
  --panel:#0b131c;
  --panel2:#0a1118;

  --text:#e7f0f6;          /* brighter */
  --text2:#d2e1ea;
  --muted:#9fb2bf;         /* brighter muted */
  --muted2:#7f94a3;

  --cyan:#00E0FF;
  --cyan2:#58f3ff;

  --green:#3CFF98;
  --amber:#FFCA28;
  --orange:#FF8A3D;
  --red:#FF3B5C;
  --black:#A7B7C4;

  --border: rgba(255,255,255,0.085);  /* stronger border */
  --border2: rgba(0,224,255,0.10);

  --shadow: rgba(0,0,0,0.55);
}

html, body {
  background: radial-gradient(1200px 800px at 18% 12%, #071826 0%, var(--bg0) 44%, #030407 100%) !important;
  color: var(--text) !important;
}

/* Keep app content above particles */
.stApp {
  position: relative;
  z-index: 1;
}

/* Global type */
* { -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale; }
a, a:visited { color: var(--cyan2) !important; }

/* Header */
.fusion-title{
  font-size: 34px;
  font-weight: 900;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--cyan);
  margin: 6px 0 0 0;
}
.fusion-subtitle{
  font-size: 13px;
  color: var(--muted);
  letter-spacing: 0.08em;
  margin: -2px 0 10px 0;
}

/* Cards */
.ops-card{
  background: linear-gradient(180deg, rgba(11,19,28,0.86), rgba(9,14,20,0.84));
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 12px 14px;
  box-shadow: 0 12px 36px var(--shadow);
}
.ops-label{
  font-size: 11px;
  color: var(--muted);
  letter-spacing: 0.16em;
  text-transform: uppercase;
  margin-bottom: 8px;
}
.ops-mono{
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
}

/* Status bar */
.statusbar{
  display:flex;
  flex-wrap: wrap;
  gap: 10px;
  align-items:center;
  justify-content: space-between;
  padding: 10px 12px;
  border-radius: 14px;
  border: 1px solid var(--border);
  background: rgba(7,10,14,0.78);
}
.sb-item{ font-size: 12px; color: var(--text); opacity: 0.98; }
.sb-k{ color: var(--muted); margin-right: 6px; }
.sb-v b{ color: var(--cyan2); }

.pill{
  display:inline-block;
  padding: 2px 9px;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: rgba(255,255,255,0.04);
  font-size: 11px;
  letter-spacing: 0.10em;
  text-transform: uppercase;
}
.pill.green{ color: var(--green); border-color: rgba(60,255,152,0.25); }
.pill.amber{ color: var(--amber); border-color: rgba(255,202,40,0.25); }
.pill.orange{ color: var(--orange); border-color: rgba(255,138,61,0.25); }
.pill.red{ color: var(--red); border-color: rgba(255,59,92,0.25); }
.pill.black{ color: var(--black); border-color: rgba(167,183,196,0.25); }

@keyframes flash-bg {
  0% { background: rgba(7,10,14,0.78); }
  45% { background: rgba(255,59,92,0.18); }
  100% { background: rgba(7,10,14,0.78); }
}
.flash{ animation: flash-bg 0.9s ease-in-out 2; }

@keyframes pulse-ring {
  0% { box-shadow: 0 0 0 0 rgba(255,59,92,0.60); }
  100% { box-shadow: 0 0 0 12px rgba(255,59,92,0); }
}
.critical-dot{
  width: 10px; height: 10px; border-radius: 8px;
  display:inline-block; background: var(--red);
  margin-right: 8px; animation: pulse-ring 1.2s ease-out infinite;
}

small { color: var(--muted) !important; }

/* Sidebar */
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(6,10,16,0.92), rgba(4,6,10,0.90));
  border-right: 1px solid var(--border);
}

/* Streamlit widget legibility patch */
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] span,
div[data-testid="stMarkdownContainer"] li { color: var(--text) !important; }

label, .stTextInput label, .stTextArea label, .stSelectbox label,
.stRadio label, .stCheckbox label, .stSlider label {
  color: var(--text2) !important;
  font-weight: 650 !important;
  letter-spacing: 0.02em;
}

input, textarea {
  color: var(--text) !important;
  background: rgba(255,255,255,0.04) !important;
}

div[data-baseweb="select"] > div {
  background: rgba(255,255,255,0.04) !important;
  color: var(--text) !important;
  border-color: var(--border) !important;
}

div[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 12px !important; overflow: hidden; }
div[data-testid="stTable"] { border: 1px solid var(--border) !important; border-radius: 12px !important; overflow: hidden; }

button[kind="primary"], button[kind="secondary"]{
  border-radius: 12px !important;
}

div[data-testid="stToolbar"]{ opacity:0.18; }
</style>
"""
st.markdown(OPS_CSS, unsafe_allow_html=True)

# ================================================================
# Particles background (subtle)
# ================================================================

def render_particles_background(enabled: bool):
    if not enabled:
        return
    html = """
    <div id="tsparticles" style="
      position: fixed; inset: 0;
      z-index: 0;
      pointer-events: none;
      opacity: 0.45;">
    </div>
    <script src="https://cdn.jsdelivr.net/npm/tsparticles@3/tsparticles.bundle.min.js"></script>
    <script>
      (async () => {
        if (window.__fusionParticlesInit) return;
        window.__fusionParticlesInit = true;
        await tsParticles.load({
          id: "tsparticles",
          options: {
            background: { color: { value: "transparent" }},
            fpsLimit: 60,
            particles: {
              number: { value: 88, density: { enable: true, area: 980 }},
              color: { value: ["#00E0FF", "#58f3ff", "#9fb2bf"] },
              links: { enable: true, distance: 130, color: "#00E0FF", opacity: 0.12, width: 1 },
              move: { enable: true, speed: 0.55, outModes: { default: "out" } },
              opacity: { value: { min: 0.10, max: 0.40 } },
              size: { value: { min: 0.8, max: 2.3 } }
            },
            interactivity: { events: { resize: true } },
            detectRetina: true
          }
        });
      })();
    </script>
    """
    components.html(html, height=0)

# ================================================================
# Helpers: canonicalization + stable IDs
# ================================================================

def _round_floats(x: Any, ndigits: int = 6) -> Any:
    if isinstance(x, float):
        if math.isnan(x):
            return "NaN"
        if math.isinf(x):
            return "Inf" if x > 0 else "-Inf"
        return round(x, ndigits)
    if isinstance(x, (list, tuple)):
        return [_round_floats(v, ndigits) for v in x]
    if isinstance(x, dict):
        return {str(k): _round_floats(v, ndigits) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    return x

def canonical_json(obj: Any) -> str:
    return json.dumps(
        _round_floats(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )

def sha256_hex(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()

def stable_id(prefix: str, payload: Any) -> str:
    h = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{h}"

# ================================================================
# Threat Doctrine (V8)
# ================================================================

class ThreatLevel(str, Enum):
    GREEN = "GREEN"     # normal
    AMBER = "AMBER"     # elevated
    ORANGE = "ORANGE"   # severe
    RED = "RED"         # critical
    BLACK = "BLACK"     # unsafe / autonomous stop

THREAT_COLOR_CLASS = {
    ThreatLevel.GREEN: "green",
    ThreatLevel.AMBER: "amber",
    ThreatLevel.ORANGE: "orange",
    ThreatLevel.RED: "red",
    ThreatLevel.BLACK: "black",
}

THREAT_DOCTRINE = [
    {
        "level": ThreatLevel.GREEN,
        "rule": "Integrity ≥ 0.80 and CriticalInsights == 0",
        "meaning": "Nominal. Autonomy permitted within policy.",
        "ops": "Monitor. No special actions required.",
    },
    {
        "level": ThreatLevel.AMBER,
        "rule": "0.65 ≤ Integrity < 0.80 OR (CriticalInsights == 0 and Warnings elevated)",
        "meaning": "Elevated monitoring. Conditions trending.",
        "ops": "Increase sampling. Validate feeds. Prep human gate.",
    },
    {
        "level": ThreatLevel.ORANGE,
        "rule": "0.45 ≤ Integrity < 0.65 OR CriticalInsights in [1..2] OR QuarantineRate high",
        "meaning": "Severe. Human must actively supervise.",
        "ops": "Require approval for side effects. Investigate anomaly source.",
    },
    {
        "level": ThreatLevel.RED,
        "rule": "0.30 ≤ Integrity < 0.45 OR CriticalInsights ≥ 3",
        "meaning": "Critical. High risk of bad actions.",
        "ops": "Block auto-actions. Force operator acknowledgements. Consider manual mode.",
    },
    {
        "level": ThreatLevel.BLACK,
        "rule": "Integrity < 0.30 OR data integrity compromised",
        "meaning": "Unsafe. Autonomy halted by doctrine.",
        "ops": "Stop autonomous side effects. Snapshot, contain, audit, recover.",
    },
]

# ================================================================
# Data Models
# ================================================================

class ValidationDisposition(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    QUARANTINED = "quarantined"

@dataclass
class SourceRuntime:
    id: str
    name: str
    enabled: bool = True
    operator_weight: float = 1.0
    manual_mode: bool = False

@dataclass(frozen=True)
class EventEnvelope:
    event_id: str
    source_id: str
    ingest_tick: int
    event_ts: float
    schema_version: str
    payload: Dict[str, Any]

@dataclass
class QuarantineEntry:
    envelope: EventEnvelope
    disposition: ValidationDisposition
    reason: str

@dataclass
class Insight:
    insight_id: str
    source_id: str
    tick: int
    level: Literal["Info", "Warning", "Critical"]
    kind: Literal["clarity", "trust", "correlation", "schema", "threat"]
    msg: str
    metrics: Dict[str, float]

@dataclass
class Recommendation:
    rec_id: str
    insight_id: str
    tick: int
    level: Literal["Info", "Warning", "Critical"]
    action: str
    action_class: str
    target_source_id: str
    rationale: str
    confidence: float
    requires_approval: bool
    blocked_reason: str = ""
    integrity_at_issue: float = 1.0
    operator_load_at_issue: str = "Low"
    threat_at_issue: str = "GREEN"

@dataclass
class Decision:
    decision_id: str
    tick: int
    rec_id: str
    choice: Literal["approve", "reject", "defer"]
    comment: str

@dataclass
class AutonomyAction:
    action_id: str
    tick: int
    rec_id: str
    action_class: str
    target_source_id: str
    status: Literal["queued", "executed", "blocked"]
    note: str = ""

@dataclass
class TrustComponents:
    freshness: float
    schema: float
    dropout: float
    drift: float
    manual_penalty: float
    operator_weight: float

@dataclass
class TrustSample:
    tick: int
    total: float
    base: float
    components: TrustComponents
    deltas: Optional[TrustComponents] = None
    blame: List[str] = field(default_factory=list)

@dataclass
class SourceStats:
    attempts: int = 0
    accepted: int = 0
    rejected: int = 0
    quarantined: int = 0
    last_delivery_tick: Optional[int] = None
    watermark_ts: Optional[float] = None
    enabled_ticks: int = 0
    ticks_with_delivery: int = 0
    recent_lags: List[float] = field(default_factory=list)

@dataclass
class PolicyConfig:
    integrity_floor: float = 0.30
    approval_required_below: float = 0.65
    approval_required_for: Set[str] = field(default_factory=lambda: {"escalate", "cross_feed", "set_manual", "schema_investigate"})
    always_requires_approval: Set[str] = field(default_factory=lambda: {"set_manual"})
    pending_med: int = 3
    pending_high: int = 6
    suppress_warnings_under_high_load: bool = True

@dataclass
class InjectionConfig:
    # Journaled knobs for demo realism (replayable)
    dropout_rate: float = 0.0          # [0..1] chance to omit an event from a source
    spoof_jitter: float = 0.0          # [0..1] increases heading/speed noise
    schema_corrupt_rate: float = 0.0   # [0..1] chance to corrupt payload shape
    incident_spike_rate: float = 0.0   # [0..1] increases incident severity
    storm_mode: bool = False           # affects weather + coherence

@dataclass
class SystemState:
    tick: int = 0
    sources: Dict[str, SourceRuntime] = field(default_factory=dict)
    events: Dict[str, List[EventEnvelope]] = field(default_factory=dict)
    quarantine: List[QuarantineEntry] = field(default_factory=list)
    stats: Dict[str, SourceStats] = field(default_factory=dict)
    metrics_history: Dict[str, List[Dict[str, float]]] = field(default_factory=dict)
    trust_history: Dict[str, List[TrustSample]] = field(default_factory=dict)
    insights: List[Insight] = field(default_factory=list)
    recommendations: List[Recommendation] = field(default_factory=list)
    autonomy_actions: List[AutonomyAction] = field(default_factory=list)
    decisions: Dict[str, Decision] = field(default_factory=dict)
    last_insight_tick: Dict[str, int] = field(default_factory=dict)
    last_critical_count: int = 0
    critical_flash: bool = False
    seen_event_ids: Set[str] = field(default_factory=set)
    injections: InjectionConfig = field(default_factory=InjectionConfig)

# ================================================================
# Engine Config
# ================================================================

DEFAULT_CLARITY_CONFIG = {
    "window_size": 12,
    "vol_warning": 0.35,
    "vol_critical": 0.60,
    "inst_warning": 0.40,
    "inst_critical": 0.65,
    "nov_warning": 0.50,
    "nov_critical": 0.75,
}
INSIGHT_REFRACTORY_TICKS = 5

DEFAULT_TRUST_WEIGHTS = {
    "freshness": 0.30,
    "schema": 0.25,
    "dropout": 0.25,
    "drift": 0.20,
}

DEFAULT_POLICY = PolicyConfig()

MAX_EVENTS_PER_SOURCE = 600
MAX_INSIGHTS = 2600
MAX_RECS = 2400
MAX_QUARANTINE = 2400
MAX_ACTIONS = 2400

# ================================================================
# Schema validation (minimal hardened)
# ================================================================

def validate_event(envelope: EventEnvelope) -> Tuple[ValidationDisposition, Dict[str, Any], str]:
    p = envelope.payload
    if not isinstance(p, dict):
        return (ValidationDisposition.REJECTED, {}, "payload not dict")

    sid = envelope.source_id
    sv = envelope.schema_version
    if sv != "v1":
        return (ValidationDisposition.QUARANTINED, p, f"unknown schema version {sv}")

    def has_num(k: str) -> bool:
        return k in p and isinstance(p[k], (int, float)) and not (
            isinstance(p[k], float) and (math.isnan(p[k]) or math.isinf(p[k]))
        )

    if sid == "ship":
        req = ["lat", "lon", "speed", "heading"]
        if not all(has_num(k) for k in req):
            return (ValidationDisposition.REJECTED, p, "ship missing/invalid fields")
        if not (-90 <= float(p["lat"]) <= 90 and -180 <= float(p["lon"]) <= 180):
            return (ValidationDisposition.QUARANTINED, p, "ship lat/lon out of bounds")
        return (ValidationDisposition.ACCEPTED, p, "ok")

    if sid == "air":
        req = ["lat", "lon", "alt", "speed", "heading"]
        if not all(has_num(k) for k in req):
            return (ValidationDisposition.REJECTED, p, "air missing/invalid fields")
        if not (-90 <= float(p["lat"]) <= 90 and -180 <= float(p["lon"]) <= 180):
            return (ValidationDisposition.QUARANTINED, p, "air lat/lon out of bounds")
        return (ValidationDisposition.ACCEPTED, p, "ok")

    if sid == "incident":
        req = ["value", "severity"]
        if not (has_num("value") and has_num("severity")):
            return (ValidationDisposition.REJECTED, p, "incident missing/invalid fields")
        return (ValidationDisposition.ACCEPTED, p, "ok")

    if sid == "weather":
        req = ["wind", "value"]
        if not all(has_num(k) for k in req):
            return (ValidationDisposition.REJECTED, p, "weather missing/invalid fields")
        return (ValidationDisposition.ACCEPTED, p, "ok")

    return (ValidationDisposition.QUARANTINED, p, f"unknown source_id {sid}")

# ================================================================
# Clarity Engine
# ================================================================

class ClarityEngine:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @staticmethod
    def _volatility(events: List[EventEnvelope]) -> float:
        if len(events) < 4:
            return 0.0
        speeds = [float(e.payload.get("speed", 0.0)) for e in events]
        return float(min(1.0, float(np.std(speeds)) / 25.0))

    @staticmethod
    def _instability(events: List[EventEnvelope]) -> float:
        if len(events) < 4:
            return 0.0
        headings = [float(e.payload.get("heading", 0.0)) for e in events]
        diffs = np.abs(np.diff(headings))
        return float(min(1.0, float(np.mean(diffs)) / 45.0))

    @staticmethod
    def _novelty(events: List[EventEnvelope]) -> float:
        if len(events) < 4:
            return 0.0
        vals = []
        for e in events:
            vals.append(float(e.payload.get("value", e.payload.get("speed", 0.0))))
        if len(set(vals)) == 1:
            return 0.0
        return float(min(1.0, abs(vals[-1] - float(np.mean(vals))) / (float(np.std(vals)) + 1e-5)))

    def compute_metrics(self, events: List[EventEnvelope]) -> Dict[str, float]:
        tail = events[-self.config["window_size"]:]
        vol = self._volatility(tail)
        inst = self._instability(tail)
        nov = self._novelty(tail)
        clarity = float(1.0 - (0.4 * vol + 0.4 * inst + 0.2 * nov))
        return {"vol": float(vol), "inst": float(inst), "nov": float(nov), "clarity": float(np.clip(clarity, 0.0, 1.0))}

    def assess_level(self, metrics: Dict[str, float]) -> Optional[str]:
        vol, inst, nov = metrics["vol"], metrics["inst"], metrics["nov"]
        critical = (vol >= self.config["vol_critical"]) or (inst >= self.config["inst_critical"]) or (nov >= self.config["nov_critical"])
        warning = (vol >= self.config["vol_warning"]) or (inst >= self.config["inst_warning"]) or (nov >= self.config["nov_warning"])
        if critical:
            return "Critical"
        if warning:
            return "Warning"
        return None

# ================================================================
# Trust Fabric (explainable + blame)
# ================================================================

def _extract_signal(payload: Dict[str, Any]) -> float:
    if "speed" in payload:
        return float(payload.get("speed", 0.0))
    if "value" in payload:
        return float(payload.get("value", 0.0))
    return 0.0

class TrustFabric:
    def __init__(self, weights: Dict[str, float], freshness_horizon_ticks: int = 10):
        self.w = weights
        self.freshness_horizon = max(3, int(freshness_horizon_ticks))

    @staticmethod
    def _drift_score(events: List[EventEnvelope]) -> float:
        if len(events) < 10:
            return 0.0
        vals = [_extract_signal(e.payload) for e in events[-30:]]
        xs = np.arange(len(vals))
        slope, _ = np.polyfit(xs, vals, 1)
        slope_component = min(1.0, abs(float(slope)) / 5.0)

        diffs = np.diff(vals)
        if len(diffs) < 3 or float(np.std(diffs)) < 1e-5:
            z_component = 0.0
        else:
            z = (float(diffs[-1]) - float(np.mean(diffs))) / (float(np.std(diffs)) + 1e-5)
            z_component = min(1.0, abs(float(z)) / 3.0)

        return float(np.clip(0.5 * slope_component + 0.5 * z_component, 0.0, 1.0))

    def compute_sample(
        self,
        tick: int,
        src: SourceRuntime,
        stats: SourceStats,
        accepted_events: List[EventEnvelope],
        prev: Optional["TrustSample"],
    ) -> "TrustSample":
        if stats.last_delivery_tick is None:
            freshness = 0.0
        else:
            delay = tick - stats.last_delivery_tick
            freshness = float(np.clip(1.0 - (delay / float(self.freshness_horizon)), 0.0, 1.0))

        schema = float(np.clip(stats.accepted / float(stats.attempts), 0.0, 1.0)) if stats.attempts > 0 else 1.0
        dropout = float(np.clip(stats.ticks_with_delivery / float(stats.enabled_ticks), 0.0, 1.0)) if stats.enabled_ticks > 0 else 1.0
        drift = float(np.clip(self._drift_score(accepted_events), 0.0, 1.0))

        base = (
            self.w["freshness"] * freshness
            + self.w["schema"] * schema
            + self.w["dropout"] * dropout
            + self.w["drift"] * (1.0 - drift)
        )
        base = float(np.clip(base, 0.0, 1.0))

        manual_penalty = 0.4 if src.manual_mode else 1.0
        operator_weight = float(np.clip(src.operator_weight, 0.0, 1.0))
        total = float(np.clip(base * manual_penalty * operator_weight, 0.0, 1.0))

        comp = TrustComponents(
            freshness=float(freshness),
            schema=float(schema),
            dropout=float(dropout),
            drift=float(drift),
            manual_penalty=float(manual_penalty),
            operator_weight=float(operator_weight),
        )

        deltas = None
        blame: List[str] = []
        if prev is not None:
            deltas = TrustComponents(
                freshness=float(comp.freshness - prev.components.freshness),
                schema=float(comp.schema - prev.components.schema),
                dropout=float(comp.dropout - prev.components.dropout),
                drift=float(comp.drift - prev.components.drift),
                manual_penalty=float(comp.manual_penalty - prev.components.manual_penalty),
                operator_weight=float(comp.operator_weight - prev.components.operator_weight),
            )
            if deltas.freshness <= -0.2:
                blame.append(f"Freshness fell {prev.components.freshness:.2f} → {comp.freshness:.2f}")
            if deltas.schema <= -0.1:
                blame.append(f"Schema score fell {prev.components.schema:.2f} → {comp.schema:.2f}")
            if deltas.dropout <= -0.1:
                blame.append(f"Dropout worsened {prev.components.dropout:.2f} → {comp.dropout:.2f}")
            if deltas.drift >= 0.2:
                blame.append(f"Drift increased {prev.components.drift:.2f} → {comp.drift:.2f}")
            if prev.components.manual_penalty != comp.manual_penalty:
                blame.append("Manual mode changed")
            if abs(deltas.operator_weight) >= 0.2:
                blame.append(f"Operator weight changed {prev.components.operator_weight:.1f} → {comp.operator_weight:.1f}")

        return TrustSample(tick=tick, total=total, base=base, components=comp, deltas=deltas, blame=blame)

    @staticmethod
    def generate_trust_insight(sid: str, sample: TrustSample, src: SourceRuntime) -> Optional[Insight]:
        t = sample.total
        manual_suffix = " [manual]" if src.manual_mode else ""
        if t >= 0.8 and sample.tick % 30 == 0:
            level = "Info"
            msg = f"{sid.upper()} trust healthy{manual_suffix}"
        elif t < 0.4:
            level = "Critical"
            msg = f"{sid.upper()} trust drop — data quality risk{manual_suffix}"
        elif t < 0.6:
            level = "Warning"
            msg = f"{sid.upper()} trust degradation — monitor feed{manual_suffix}"
        else:
            return None

        metrics = {
            "trust": float(sample.total),
            "base": float(sample.base),
            "freshness": float(sample.components.freshness),
            "schema": float(sample.components.schema),
            "dropout": float(sample.components.dropout),
            "drift": float(sample.components.drift),
            "manual_penalty": float(sample.components.manual_penalty),
            "operator_weight": float(sample.components.operator_weight),
        }

        return Insight(
            insight_id=stable_id("ins_trust", {"sid": sid, "tick": sample.tick, "level": level, "metrics": metrics}),
            source_id=sid,
            tick=sample.tick,
            level=level,  # type: ignore
            kind="trust",
            msg=msg,
            metrics=metrics,
        )

# ================================================================
# Journal: tamper-evident hash chain + optional HMAC
# ================================================================

def _journal_entry_hash(prev_hash: str, entry: Dict[str, Any]) -> str:
    material = canonical_json({
        "prev": prev_hash,
        "seq": entry["seq"],
        "tick": entry["tick"],
        "kind": entry["kind"],
        "payload": entry["payload"],
    })
    return hashlib.sha256(material.encode("utf-8")).hexdigest()

def _journal_entry_sig(entry_hash: str) -> str:
    secret = os.getenv("FUSIONOPS_JOURNAL_SECRET", "")
    if not secret:
        return ""
    return hmac.new(secret.encode("utf-8"), entry_hash.encode("utf-8"), hashlib.sha256).hexdigest()

def validate_journal(journal: List[Dict[str, Any]]) -> Tuple[bool, str]:
    prev = "GENESIS"
    secret = os.getenv("FUSIONOPS_JOURNAL_SECRET", "")
    for i, e in enumerate(journal):
        if int(e.get("seq", -1)) != i:
            return False, f"seq mismatch at {i}"
        expected = _journal_entry_hash(prev, {"seq": e["seq"], "tick": e["tick"], "kind": e["kind"], "payload": e["payload"]})
        if e.get("hash") != expected:
            return False, f"hash mismatch at {i}"
        if secret:
            expected_sig = hmac.new(secret.encode("utf-8"), expected.encode("utf-8"), hashlib.sha256).hexdigest()
            if e.get("sig") != expected_sig:
                return False, f"sig mismatch at {i}"
        prev = e["hash"]
    return True, "ok"

def validate_journal_semantics(journal: List[Dict[str, Any]]) -> Tuple[bool, str]:
    last_tick = -1
    seen_tick_batches: Set[int] = set()
    for e in journal:
        tick = int(e.get("tick", 0))
        kind = str(e.get("kind", ""))
        if tick < 0:
            return False, "negative tick"
        if tick < last_tick:
            return False, f"tick decreased at seq {e['seq']}"
        last_tick = tick

        if kind == "tick_batch":
            if tick in seen_tick_batches:
                return False, f"duplicate tick_batch tick={tick}"
            seen_tick_batches.add(tick)
            if "events" not in e.get("payload", {}):
                return False, f"tick_batch missing events at tick={tick}"

        if kind == "decision":
            if "rec_id" not in e.get("payload", {}):
                return False, "decision missing rec_id"
            if "choice" not in e.get("payload", {}):
                return False, "decision missing choice"

        if kind == "operator_cmd":
            if "cmd" not in e.get("payload", {}):
                return False, "operator_cmd missing cmd"

    return True, "ok"

def append_journal(kind: str, tick: int, payload: Dict[str, Any]):
    j = st.session_state.journal
    seq = len(j)
    prev_hash = j[-1].get("hash", "GENESIS") if j else "GENESIS"
    entry = {"seq": seq, "kind": kind, "tick": int(tick), "payload": payload}
    entry_hash = _journal_entry_hash(prev_hash, entry)
    entry["hash"] = entry_hash
    sig = _journal_entry_sig(entry_hash)
    if sig:
        entry["sig"] = sig
    j.append(entry)

def _iter_journal_until(journal: List[Dict[str, Any]], until_tick: Optional[int]) -> Iterable[Dict[str, Any]]:
    for e in journal:
        if until_tick is not None and int(e.get("tick", 0)) > int(until_tick):
            break
        yield e

# ================================================================
# Synthetic telemetry generation (with journaled injectors)
# ================================================================

def _jitter(seed: int, tick: int, sid: str) -> float:
    key = sha256_hex({"seed": seed, "tick": tick, "sid": sid})
    x = int(key[:8], 16) / float(16**8)
    j = (x - 0.5) * 3.0
    if tick % 17 == 0:
        j -= 2.0
    return float(j)

def _rand01(seed: int, tick: int, key: str) -> float:
    h = sha256_hex({"seed": seed, "tick": tick, "key": key})
    return int(h[:8], 16) / float(16**8)

def _maybe_corrupt_payload(seed: int, tick: int, sid: str, p: Dict[str, Any], rate: float) -> Dict[str, Any]:
    if rate <= 0:
        return p
    r = _rand01(seed, tick, f"corrupt:{sid}")
    if r > rate:
        return p
    # Deterministic corruption variants
    mode = int(_rand01(seed, tick, f"corruptmode:{sid}") * 3.999)
    if mode == 0:
        return {"oops": "badshape", "payload": p}  # wrong shape
    if mode == 1:
        q = dict(p)
        q.pop(next(iter(q.keys())), None)          # missing required field
        return q
    if mode == 2:
        q = dict(p)
        q[next(iter(q.keys()))] = "NaN"           # wrong type
        return q
    q = dict(p)
    q["lat"] = 9999                              # out of bounds
    return q

def generate_synthetic_batch(
    tick: int,
    sources: Dict[str, SourceRuntime],
    run_cfg: Dict[str, Any],
    inj: InjectionConfig,
) -> List[EventEnvelope]:
    seed = int(run_cfg["seed"])
    dt = float(run_cfg.get("tick_dt_s", 1.0))
    logical_ts = tick * dt
    out: List[EventEnvelope] = []

    def mk_event(source_id: str, schema_version: str, payload: Dict[str, Any]) -> EventEnvelope:
        event_ts = float(logical_ts + _jitter(seed, tick, source_id))
        eid = stable_id("evt", {"tick": tick, "sid": source_id, "ts": event_ts, "payload": payload})
        return EventEnvelope(
            event_id=eid,
            source_id=source_id,
            ingest_tick=tick,
            event_ts=event_ts,
            schema_version=schema_version,
            payload=payload,
        )

    def drop_this(sid: str) -> bool:
        if inj.dropout_rate <= 0:
            return False
        return _rand01(seed, tick, f"drop:{sid}") < inj.dropout_rate

    # Storm mode makes weather rougher and coherence worse (ships/air drift)
    storm = bool(inj.storm_mode)

    # --- SHIP ---
    if sources["ship"].enabled and not drop_this("ship"):
        base_lat = 40.0 + float(np.sin(tick / (9.0 if not storm else 7.0)))
        base_lon = -80.0 + float(np.cos(tick / (9.0 if not storm else 7.0)))
        base_speed = 11.0 + float(np.sin(tick / (5.0 if not storm else 4.0)) * (3.0 if not storm else 4.5))
        base_heading = float((tick * 4) % 360)

        # spoof jitter increases heading/speed noise deterministically
        sj = float(np.clip(inj.spoof_jitter, 0.0, 1.0))
        if sj > 0:
            noise = (_rand01(seed, tick, "shipnoise") - 0.5) * 2.0
            base_heading = float((base_heading + noise * 90.0 * sj) % 360)
            base_speed = float(max(0.0, base_speed + noise * 6.0 * sj))

        payload = {"lat": base_lat, "lon": base_lon, "speed": base_speed, "heading": base_heading}
        payload = _maybe_corrupt_payload(seed, tick, "ship", payload, inj.schema_corrupt_rate)
        out.append(mk_event("ship", run_cfg["sources_default"]["ship"]["schema_version"], payload))

    # --- AIR ---
    if sources["air"].enabled and not drop_this("air"):
        base_lat = 39.0 + float(np.sin(tick / (12.0 if not storm else 9.0)))
        base_lon = -81.0 + float(np.cos(tick / (12.0 if not storm else 9.0)))
        base_alt = 31000.0 + float(np.sin(tick / 6.0) * (1500.0 if not storm else 2400.0))
        base_speed = 440.0 + float(np.cos(tick / 7.0) * (40.0 if not storm else 70.0))
        base_heading = float((tick * 6) % 360)

        sj = float(np.clip(inj.spoof_jitter, 0.0, 1.0))
        if sj > 0:
            noise = (_rand01(seed, tick, "airnoise") - 0.5) * 2.0
            base_heading = float((base_heading + noise * 120.0 * sj) % 360)
            base_speed = float(max(0.0, base_speed + noise * 55.0 * sj))
            base_alt = float(max(0.0, base_alt + noise * 2500.0 * sj))

        payload = {"lat": base_lat, "lon": base_lon, "alt": base_alt, "speed": base_speed, "heading": base_heading}
        payload = _maybe_corrupt_payload(seed, tick, "air", payload, inj.schema_corrupt_rate)
        out.append(mk_event("air", run_cfg["sources_default"]["air"]["schema_version"], payload))

    # --- INCIDENT ---
    if sources["incident"].enabled and not drop_this("incident"):
        r = _rand01(seed, tick, "incident")
        spike = float(np.clip(inj.incident_spike_rate, 0.0, 1.0))
        val = float((r - 0.5) * 0.8 + (1.0 if tick % 40 == 0 else 0.0))
        sev = 0
        if tick % 50 == 0:
            sev = 2
        elif tick % 20 == 0:
            sev = 1

        # spikes
        if spike > 0 and _rand01(seed, tick, "incident_spike") < spike:
            sev = min(5, sev + 2)
            val = float(val + 1.2)

        payload = {"value": val, "severity": int(sev)}
        payload = _maybe_corrupt_payload(seed, tick, "incident", payload, inj.schema_corrupt_rate)
        out.append(mk_event("incident", run_cfg["sources_default"]["incident"]["schema_version"], payload))

    # --- WEATHER ---
    if sources["weather"].enabled and not drop_this("weather"):
        wind = 10.0 + float(np.sin(tick / (12.0 if not storm else 8.0)) * (3.0 if not storm else 8.0))
        tempish = float(np.cos(tick / (14.0 if not storm else 10.0)) * (4.0 if not storm else 7.0))
        payload = {"wind": wind, "value": tempish}
        payload = _maybe_corrupt_payload(seed, tick, "weather", payload, inj.schema_corrupt_rate)
        out.append(mk_event("weather", run_cfg["sources_default"]["weather"]["schema_version"], payload))

    return out

# ================================================================
# Core reducers + scoring
# ================================================================

CLARITY = ClarityEngine(DEFAULT_CLARITY_CONFIG)
TRUST = TrustFabric(DEFAULT_TRUST_WEIGHTS)

def _trim_list_inplace(xs: List[Any], cap: int):
    if len(xs) > cap:
        del xs[: len(xs) - cap]

def _insight_key(ins: Insight) -> str:
    return f"{ins.kind}:{ins.source_id}:{ins.level}:{ins.msg}"

def pending_recommendations(state: SystemState) -> List[Recommendation]:
    decided = set(state.decisions.keys())
    return [r for r in state.recommendations if r.rec_id not in decided]

def operator_load_label(policy: PolicyConfig, pending_count: int, recent_insights: int) -> str:
    if pending_count >= policy.pending_high or recent_insights >= 10:
        return "High"
    if pending_count >= policy.pending_med or recent_insights >= 5:
        return "Medium"
    return "Low"

def _global_clarity_score(state: SystemState) -> float:
    vals = []
    for sid, evs in state.events.items():
        if len(evs) >= 4:
            vals.append(CLARITY.compute_metrics(evs)["clarity"])
    return float(np.clip(float(np.mean(vals)), 0.0, 1.0)) if vals else 1.0

def _global_trust_score(state: SystemState) -> float:
    vals = []
    for sid, hist in state.trust_history.items():
        if hist:
            vals.append(hist[-1].total)
    return float(np.clip(float(np.mean(vals)), 0.0, 1.0)) if vals else 1.0

def _global_health_score(state: SystemState) -> float:
    scores = []
    now_tick = state.tick
    for sid, src in state.sources.items():
        last_t = state.stats.get(sid, SourceStats()).last_delivery_tick
        if last_t is None:
            scores.append(0.2)
        else:
            delay = now_tick - last_t
            if delay <= 2:
                scores.append(1.0)
            elif delay <= 6:
                scores.append(0.6)
            else:
                scores.append(0.3)
    return float(np.clip(float(np.mean(scores)), 0.0, 1.0)) if scores else 1.0

def _global_integrity_score(state: SystemState) -> float:
    clarity = _global_clarity_score(state)
    trust = _global_trust_score(state)
    health = _global_health_score(state)
    return float(np.clip(0.4 * clarity + 0.3 * trust + 0.3 * health, 0.0, 1.0))

def integrity_state_label(score: float) -> str:
    if score >= 0.85:
        return "Stable Autonomous Operation"
    if score >= 0.7:
        return "Watch / Elevated Monitoring"
    if score >= 0.5:
        return "Degraded — Verify Feeds"
    if score >= 0.3:
        return "Critical — Operator Focus Required"
    return "Unsafe Autonomous Operation"

def _apply_critical_flash(state: SystemState):
    total_critical = sum(1 for i in state.insights if i.level == "Critical")
    state.critical_flash = total_critical > state.last_critical_count
    state.last_critical_count = total_critical

def _update_metrics_history(state: SystemState):
    for sid, evs in state.events.items():
        if not evs:
            continue
        metrics = CLARITY.compute_metrics(evs)
        hist = state.metrics_history.setdefault(sid, [])
        row = {"tick": float(state.tick)}
        row.update({k: float(v) for k, v in metrics.items()})
        hist.append(row)
        _trim_list_inplace(hist, 260)

def _update_trust_history(state: SystemState) -> List[Insight]:
    trust_insights: List[Insight] = []
    for sid, src in state.sources.items():
        stats = state.stats.setdefault(sid, SourceStats())
        evs = state.events.get(sid, [])
        prev = state.trust_history.get(sid, [])[-1] if state.trust_history.get(sid) else None
        sample = TRUST.compute_sample(state.tick, src, stats, evs, prev)
        hist = state.trust_history.setdefault(sid, [])
        hist.append(sample)
        _trim_list_inplace(hist, 260)
        ins = TRUST.generate_trust_insight(sid, sample, src)
        if ins is not None:
            trust_insights.append(ins)
    return trust_insights

def _filter_new_insights(state: SystemState, candidates: List[Insight]) -> List[Insight]:
    out: List[Insight] = []
    now_tick = state.tick
    for ins in candidates:
        key = _insight_key(ins)
        last = state.last_insight_tick.get(key)
        if last is not None and (now_tick - last) < INSIGHT_REFRACTORY_TICKS:
            continue
        out.append(ins)
        state.last_insight_tick[key] = now_tick
    return out

def _generate_clarity_insights(state: SystemState) -> List[Insight]:
    out: List[Insight] = []
    for sid in sorted(state.events.keys()):
        evs = state.events.get(sid, [])
        if len(evs) < 4:
            continue
        metrics = CLARITY.compute_metrics(evs)
        level = CLARITY.assess_level(metrics)
        if level is None:
            continue
        src = state.sources.get(sid)
        if src and src.manual_mode and level in ("Critical", "Warning"):
            level2: Literal["Info", "Warning", "Critical"] = "Info"
            msg = f"{sid.upper()} stability deviation (manual mode)"
        else:
            level2 = level  # type: ignore
            msg = f"{sid.upper()} stability deviation"
        out.append(
            Insight(
                insight_id=stable_id("ins_clarity", {"sid": sid, "tick": state.tick, "level": level2, "metrics": metrics}),
                source_id=sid,
                tick=state.tick,
                level=level2,  # type: ignore
                kind="clarity",
                msg=msg,
                metrics={k: float(v) for k, v in metrics.items()},
            )
        )
    return out

def _quarantine_rate_global(state: SystemState) -> float:
    attempts = 0
    quarantined = 0
    for s in state.stats.values():
        attempts += int(s.attempts)
        quarantined += int(s.quarantined)
    return float(quarantined / float(max(1, attempts)))

def compute_threat_level(state: SystemState) -> ThreatLevel:
    integrity = _global_integrity_score(state)
    crit = sum(1 for i in state.insights[-60:] if i.level == "Critical")
    warn = sum(1 for i in state.insights[-60:] if i.level == "Warning")
    q_rate = _quarantine_rate_global(state)

    # BLACK: unsafe autonomy
    if integrity < 0.30:
        return ThreatLevel.BLACK

    # RED: critical
    if integrity < 0.45 or crit >= 3:
        return ThreatLevel.RED

    # ORANGE: severe
    if integrity < 0.65 or (1 <= crit <= 2) or q_rate > 0.18:
        return ThreatLevel.ORANGE

    # AMBER: elevated
    if integrity < 0.80 or warn >= 6 or q_rate > 0.10:
        return ThreatLevel.AMBER

    return ThreatLevel.GREEN

def _generate_threat_insight(state: SystemState, threat: ThreatLevel) -> Optional[Insight]:
    # Emit occasional doctrine insight (not every tick)
    if state.tick % 10 != 0:
        return None
    integrity = _global_integrity_score(state)
    q = _quarantine_rate_global(state)
    msg = f"Threat doctrine: {threat.value}"
    return Insight(
        insight_id=stable_id("ins_threat", {"tick": state.tick, "threat": threat.value, "integrity": integrity, "q": q}),
        source_id="global",
        tick=state.tick,
        level="Warning" if threat in (ThreatLevel.AMBER, ThreatLevel.ORANGE) else ("Critical" if threat in (ThreatLevel.RED, ThreatLevel.BLACK) else "Info"),
        kind="threat",
        msg=msg,
        metrics={"integrity": float(integrity), "quarantine_rate": float(q), "threat": float(["GREEN","AMBER","ORANGE","RED","BLACK"].index(threat.value))},
    )

def _generate_recommendations(state: SystemState, new_insights: List[Insight], policy: PolicyConfig, threat: ThreatLevel):
    pending = len(pending_recommendations(state))
    recent_ins = len(state.insights[-12:])
    load = operator_load_label(policy, pending, recent_ins)
    integrity = _global_integrity_score(state)

    # Autonomy doctrine gating based on threat
    doctrine_blocks_auto = threat in (ThreatLevel.RED, ThreatLevel.BLACK)
    doctrine_requires_gate = threat in (ThreatLevel.ORANGE, ThreatLevel.RED, ThreatLevel.BLACK)

    auto_action_classes: Set[str] = {"escalate", "schema_investigate"}  # only safe-ish demos
    side_effecting: Set[str] = {"escalate", "cross_feed", "schema_investigate", "set_manual"}

    def _record_action(rec: Recommendation):
        if rec.action_class not in auto_action_classes and rec.action_class not in policy.always_requires_approval:
            return
        status: Literal["queued", "executed", "blocked"] = "queued" if rec.requires_approval else "executed"
        note = rec.blocked_reason or ("Awaiting operator decision" if rec.requires_approval else "Auto-executed (policy allowed)")
        act_id = stable_id("act", {"tick": rec.tick, "rec_id": rec.rec_id, "status": status})
        state.autonomy_actions.append(
            AutonomyAction(
                action_id=act_id,
                tick=rec.tick,
                rec_id=rec.rec_id,
                action_class=rec.action_class,
                target_source_id=rec.target_source_id,
                status=status,
                note=note,
            )
        )
        _trim_list_inplace(state.autonomy_actions, MAX_ACTIONS)

    for ins in new_insights:
        if load == "High" and policy.suppress_warnings_under_high_load and ins.level == "Warning":
            continue

        src_cfg = state.sources.get(ins.source_id)
        manual = src_cfg.manual_mode if src_cfg else False

        action = "Log insight"
        rationale_parts: List[str] = []
        action_class = "log"
        target_sid = ins.source_id

        if ins.kind == "clarity":
            if manual:
                action = "Operator review (manual mode)"
                action_class = "manual_review"
                rationale_parts.append("Manual override active; no auto-escalation.")
            else:
                v, i, n = ins.metrics.get("vol", 0.0), ins.metrics.get("inst", 0.0), ins.metrics.get("nov", 0.0)
                cfg = CLARITY.config
                rationale_parts.append(
                    f"Clarity thresholds: vol({v:.2f}) warn≥{cfg['vol_warning']:.2f}/crit≥{cfg['vol_critical']:.2f}, "
                    f"inst({i:.2f}) warn≥{cfg['inst_warning']:.2f}/crit≥{cfg['inst_critical']:.2f}, "
                    f"nov({n:.2f}) warn≥{cfg['nov_warning']:.2f}/crit≥{cfg['nov_critical']:.2f}."
                )
                if ins.level == "Critical":
                    action = "Escalate + widen monitoring window"
                    action_class = "escalate"
                    rationale_parts.append("Critical deviation: immediate attention.")
                elif ins.level == "Warning":
                    action = "Expand monitoring and confirm pattern"
                    action_class = "monitor"
                    rationale_parts.append("Warning deviation: watch trend.")
                else:
                    action = "Acknowledge clarity state"
                    action_class = "ack"

        elif ins.kind == "trust":
            if ins.level == "Critical":
                action = "Review feed + consider manual mode"
                action_class = "set_manual"
                rationale_parts.append("Severe trust degradation indicates data quality risk.")
            elif ins.level == "Warning":
                action = "Monitor trust + review overrides"
                action_class = "monitor"
                rationale_parts.append("Trust degrading; check freshness/schema/dropout/drift.")
            else:
                action = "Acknowledge trust state"
                action_class = "ack"
                rationale_parts.append("Informational trust update.")

            th = state.trust_history.get(ins.source_id, [])
            if th and th[-1].blame:
                rationale_parts.append("Blame: " + " | ".join(th[-1].blame[:3]))

        elif ins.kind == "schema":
            if ins.level == "Critical":
                action = "Investigate schema drift + quarantine reasons"
                action_class = "schema_investigate"
                rationale_parts.append("High reject/quarantine rates detected; review contract compatibility.")
            elif ins.level == "Warning":
                action = "Review schema validation failures"
                action_class = "schema_monitor"
                rationale_parts.append("Schema issues rising; monitor quarantine lane.")
            else:
                action = "Log schema health"
                action_class = "log"

        elif ins.kind == "threat":
            action = "Acknowledge doctrine state"
            action_class = "doctrine_ack"
            rationale_parts.append("Threat doctrine updated; ensure ops posture matches.")

        requires_approval = False
        blocked_reason = ""

        # Policy always-human actions
        if action_class in policy.always_requires_approval:
            requires_approval = True
            blocked_reason = "Policy: always requires approval"

        # Doctrine gating
        if not requires_approval and doctrine_requires_gate and action_class in side_effecting:
            requires_approval = True
            blocked_reason = f"Threat doctrine {threat.value}: human gate required"

        if doctrine_blocks_auto and action_class in side_effecting:
            requires_approval = True
            blocked_reason = blocked_reason or f"Threat doctrine {threat.value}: auto side-effects blocked"

        # Integrity-based gating
        if not requires_approval:
            if integrity < policy.integrity_floor and action_class in side_effecting:
                requires_approval = True
                blocked_reason = f"Integrity {integrity:.2f} below floor {policy.integrity_floor:.2f}"
            elif integrity < policy.approval_required_below and action_class in policy.approval_required_for:
                requires_approval = True
                blocked_reason = f"Integrity {integrity:.2f} below approval threshold {policy.approval_required_below:.2f}"

        # Queue-pressure gating
        if load == "High" and action_class in policy.approval_required_for:
            requires_approval = True
            blocked_reason = blocked_reason or "Operator load High — enforce human gate"

        rationale_parts.append(f"Integrity={integrity:.2f}, OperatorLoad={load}, Threat={threat.value}.")

        # confidence: inverse of "badness"
        if "clarity" in ins.metrics:
            confidence = float(np.clip(1.0 - ins.metrics["clarity"], 0.0, 1.0))
        elif "trust" in ins.metrics:
            confidence = float(np.clip(1.0 - ins.metrics["trust"], 0.0, 1.0))
        else:
            confidence = 0.55

        rec_payload = {
            "insight_id": ins.insight_id,
            "tick": state.tick,
            "level": ins.level,
            "action": action,
            "class": action_class,
            "target": target_sid,
            "requires_approval": requires_approval,
            "integrity": float(integrity),
            "operator_load": load,
            "threat": threat.value,
        }
        rec_id = stable_id("rec", rec_payload)

        rec = Recommendation(
            rec_id=rec_id,
            insight_id=ins.insight_id,
            tick=state.tick,
            level=ins.level,
            action=action,
            action_class=action_class,
            target_source_id=target_sid,
            rationale=" ".join(rationale_parts),
            confidence=float(confidence),
            requires_approval=requires_approval,
            blocked_reason=blocked_reason,
            integrity_at_issue=float(integrity),
            operator_load_at_issue=str(load),
            threat_at_issue=threat.value,
        )
        state.recommendations.append(rec)
        _trim_list_inplace(state.recommendations, MAX_RECS)
        _record_action(rec)

def _apply_tick_batch(state: SystemState, tick: int, envelopes: List[EventEnvelope], policy: PolicyConfig, tick_dt_s: float):
    state.tick = int(tick)
    dt = float(max(1e-6, tick_dt_s))
    logical_ts = float(state.tick * dt)

    for sid, src in state.sources.items():
        stats = state.stats.setdefault(sid, SourceStats())
        if src.enabled:
            stats.enabled_ticks += 1

    delivered_this_tick: Set[str] = set()

    envelopes_sorted = sorted(envelopes, key=lambda e: (e.source_id, e.event_id))
    for env in envelopes_sorted:
        if env.event_id in state.seen_event_ids:
            state.quarantine.append(QuarantineEntry(env, ValidationDisposition.QUARANTINED, "duplicate event_id"))
            _trim_list_inplace(state.quarantine, MAX_QUARANTINE)
            continue
        state.seen_event_ids.add(env.event_id)

        sid = env.source_id
        stats = state.stats.setdefault(sid, SourceStats())
        stats.attempts += 1
        stats.last_delivery_tick = state.tick
        delivered_this_tick.add(sid)

        prior_wm = stats.watermark_ts if stats.watermark_ts is not None else env.event_ts
        stats.watermark_ts = float(max(float(prior_wm), float(env.event_ts)))

        lag = float(logical_ts - float(env.event_ts))
        stats.recent_lags.append(lag)
        _trim_list_inplace(stats.recent_lags, 140)

        disp, cleaned, reason = validate_event(env)

        if disp == ValidationDisposition.ACCEPTED:
            stats.accepted += 1
            env2 = EventEnvelope(env.event_id, env.source_id, env.ingest_tick, env.event_ts, env.schema_version, dict(cleaned))
            state.events.setdefault(sid, []).append(env2)
            _trim_list_inplace(state.events[sid], MAX_EVENTS_PER_SOURCE)
        elif disp == ValidationDisposition.QUARANTINED:
            stats.quarantined += 1
            state.quarantine.append(QuarantineEntry(env, disp, reason))
            _trim_list_inplace(state.quarantine, MAX_QUARANTINE)
        else:
            stats.rejected += 1
            state.quarantine.append(QuarantineEntry(env, disp, reason))
            _trim_list_inplace(state.quarantine, MAX_QUARANTINE)

    for sid in delivered_this_tick:
        state.stats[sid].ticks_with_delivery += 1

    _update_metrics_history(state)

    candidates: List[Insight] = []
    candidates.extend(_generate_clarity_insights(state))
    candidates.extend(_update_trust_history(state))

    # schema pulse
    if state.tick % 25 == 0:
        for sid, stats in state.stats.items():
            if stats.attempts <= 0:
                continue
            q_rate = stats.quarantined / float(max(1, stats.attempts))
            r_rate = stats.rejected / float(max(1, stats.attempts))
            level: Literal["Info", "Warning", "Critical"] = "Info"
            if r_rate > 0.05:
                level = "Warning"
            if r_rate > 0.10 or q_rate > 0.20:
                level = "Critical"
            candidates.append(
                Insight(
                    insight_id=stable_id("ins_schema", {"sid": sid, "tick": state.tick, "q": q_rate, "r": r_rate, "level": level}),
                    source_id=sid,
                    tick=state.tick,
                    level=level,
                    kind="schema",
                    msg=f"{sid.upper()} schema health — q={q_rate:.2f}, rej={r_rate:.2f}",
                    metrics={"quarantine_rate": float(q_rate), "reject_rate": float(r_rate), "attempts": float(stats.attempts)},
                )
            )

    new_insights = _filter_new_insights(state, candidates)
    state.insights.extend(new_insights)
    _trim_list_inplace(state.insights, MAX_INSIGHTS)

    # Threat doctrine insight + recommendation gating
    threat = compute_threat_level(state)
    t_ins = _generate_threat_insight(state, threat)
    if t_ins is not None:
        state.insights.append(t_ins)
        _trim_list_inplace(state.insights, MAX_INSIGHTS)

    _generate_recommendations(state, new_insights + ([t_ins] if t_ins else []), policy, threat)
    _apply_critical_flash(state)

def _apply_operator_cmd(state: SystemState, cmd: Dict[str, Any]):
    kind = cmd.get("cmd")

    if kind == "set_source":
        sid = cmd["source_id"]
        if sid not in state.sources:
            return
        src = state.sources[sid]
        if "enabled" in cmd:
            src.enabled = bool(cmd["enabled"])
        if "manual_mode" in cmd:
            src.manual_mode = bool(cmd["manual_mode"])
        if "operator_weight" in cmd:
            src.operator_weight = float(np.clip(float(cmd["operator_weight"]), 0.0, 1.0))

    elif kind == "set_injections":
        # journaled injectors
        inj = state.injections
        for k in ["dropout_rate","spoof_jitter","schema_corrupt_rate","incident_spike_rate","storm_mode"]:
            if k in cmd:
                setattr(inj, k, cmd[k])

def _apply_decision_effects(state: SystemState, decision: Decision):
    rec_map = {r.rec_id: r for r in state.recommendations}
    rec = rec_map.get(decision.rec_id)
    if rec is None:
        return

    def _set_action_status(status: Literal["queued", "executed", "blocked"], note: str):
        for act in reversed(state.autonomy_actions):
            if act.rec_id == decision.rec_id:
                act.status = status
                act.note = note
                return

    if decision.choice == "approve":
        _set_action_status("executed", f"Approved at tick {decision.tick}")
        if rec.action_class == "set_manual" and rec.target_source_id in state.sources:
            state.sources[rec.target_source_id].manual_mode = True
    elif decision.choice == "reject":
        _set_action_status("blocked", f"Rejected at tick {decision.tick}")
    else:
        _set_action_status("queued", f"Deferred at tick {decision.tick}")

def rebuild_state(journal: List[Dict[str, Any]], run_config: Dict[str, Any], until_tick: Optional[int] = None) -> SystemState:
    src_defaults = run_config["sources_default"]
    sources: Dict[str, SourceRuntime] = {
        sid: SourceRuntime(
            id=sid,
            name=cfg["name"],
            enabled=bool(cfg["enabled"]),
            operator_weight=float(cfg["operator_weight"]),
            manual_mode=bool(cfg["manual_mode"]),
        )
        for sid, cfg in src_defaults.items()
    }
    state = SystemState(tick=0, sources=sources)
    policy = PolicyConfig(**run_config.get("policy", {}))
    dt = float(run_config.get("tick_dt_s", 1.0))

    for entry in _iter_journal_until(journal, until_tick):
        kind = entry.get("kind")
        tick = int(entry.get("tick", 0))
        payload = entry.get("payload", {})

        if kind == "config_snapshot":
            continue
        if kind == "operator_cmd":
            _apply_operator_cmd(state, payload)
        elif kind == "decision":
            d = Decision(
                decision_id=stable_id("dec", payload),
                tick=tick,
                rec_id=payload["rec_id"],
                choice=payload["choice"],
                comment=payload.get("comment", ""),
            )
            state.decisions[d.rec_id] = d
            _apply_decision_effects(state, d)
        elif kind == "tick_batch":
            envs = []
            for e in payload.get("events", []):
                envs.append(EventEnvelope(
                    event_id=e["event_id"],
                    source_id=e["source_id"],
                    ingest_tick=int(e["ingest_tick"]),
                    event_ts=float(e["event_ts"]),
                    schema_version=e["schema_version"],
                    payload=dict(e["payload"]),
                ))
            _apply_tick_batch(state, tick, envs, policy, tick_dt_s=dt)

    return state

# ================================================================
# Run State Init
# ================================================================

def init_run_state():
    if st.session_state.get("initialized"):
        return

    st.session_state.initialized = True

    run_config = {
        "version": "8.0",
        "seed": int(pd.Timestamp.utcnow().value % 1_000_000),
        "clarity": DEFAULT_CLARITY_CONFIG,
        "trust_weights": DEFAULT_TRUST_WEIGHTS,
        "policy": asdict(DEFAULT_POLICY),
        "sources_default": {
            "ship": {"name": "Maritime Feed", "enabled": True, "operator_weight": 1.0, "manual_mode": False, "schema_version": "v1"},
            "air": {"name": "Aviation Feed", "enabled": True, "operator_weight": 1.0, "manual_mode": False, "schema_version": "v1"},
            "incident": {"name": "Incident Reports", "enabled": True, "operator_weight": 1.0, "manual_mode": False, "schema_version": "v1"},
            "weather": {"name": "Weather Layer", "enabled": True, "operator_weight": 1.0, "manual_mode": False, "schema_version": "v1"},
        },
        "tick_dt_s": 1.0,
    }

    st.session_state.run_config = run_config
    st.session_state.config_hash = sha256_hex(run_config)
    st.session_state.journal = []
    st.session_state.explain_freeze_tick = None
    st.session_state.last_consistency = None

    append_journal("config_snapshot", 0, {"config": run_config, "config_hash": st.session_state.config_hash})

# ================================================================
# UI: Header + Status
# ================================================================

def render_header():
    st.markdown("<div class='fusion-title'>Fusion Ops V8</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='fusion-subtitle'>Threat Doctrine · Deterministic Replay · Explainable Trust · Human-Gated Autonomy · Tactical Map Surface</div>",
        unsafe_allow_html=True,
    )

def compute_state_hash(state: SystemState) -> str:
    payload = {
        "tick": state.tick,
        "sources": {sid: asdict(src) for sid, src in state.sources.items()},
        "stats": {sid: asdict(st) for sid, st in state.stats.items()},
        "injections": asdict(state.injections),
        "insights_tail": [asdict(i) for i in state.insights[-25:]],
        "recs_tail": [asdict(r) for r in state.recommendations[-25:]],
        "decisions": {rid: asdict(d) for rid, d in state.decisions.items()},
        "seen_event_ids_n": len(state.seen_event_ids),
        "journal_len": len(st.session_state.journal),
    }
    return sha256_hex(payload)[:16]

def render_statusbar(state: SystemState, policy: PolicyConfig, journal: List[Dict[str, Any]]):
    integrity = _global_integrity_score(state)
    load = operator_load_label(policy, len(pending_recommendations(state)), len(state.insights[-12:]))
    threat = compute_threat_level(state)

    flash_cls = " flash" if state.critical_flash else ""
    load_cls = {"Low": "green", "Medium": "amber", "High": "red"}[load]
    threat_cls = THREAT_COLOR_CLASS[threat]

    journal_hash = journal[-1].get("hash", "—")[:12] if journal else "—"
    state_hash = compute_state_hash(state)

    st.markdown(
        f"""
<div class="statusbar{flash_cls}">
  <div class="sb-item"><span class="sb-k">Tick</span><span class="sb-v"><b>{state.tick}</b></span></div>
  <div class="sb-item"><span class="sb-k">Threat</span><span class="pill {threat_cls}">{threat.value}</span></div>
  <div class="sb-item"><span class="sb-k">Integrity</span><span class="sb-v"><b>{integrity:.2f}</b></span></div>
  <div class="sb-item"><span class="sb-k">State</span><span class="sb-v">{integrity_state_label(integrity)}</span></div>
  <div class="sb-item"><span class="sb-k">Load</span><span class="pill {load_cls}">{load}</span></div>
  <div class="sb-item"><span class="sb-k">Config</span><span class="sb-v ops-mono">{st.session_state.config_hash[:12]}</span></div>
  <div class="sb-item"><span class="sb-k">JournalHead</span><span class="sb-v ops-mono">{journal_hash}</span></div>
  <div class="sb-item"><span class="sb-k">StateHash</span><span class="sb-v ops-mono">{state_hash}</span></div>
</div>
""",
        unsafe_allow_html=True,
    )

# ================================================================
# Sidebar: Controls + Replay + Injectors + Visuals
# ================================================================

def render_sidebar_controls(full_state: SystemState, journal: List[Dict[str, Any]]):
    st.sidebar.markdown("## Control")

    # Live mode: auto-advance + rerun loop (demo)
    st.sidebar.markdown("### Live Ops")
    auto_run = st.sidebar.toggle("Auto-run", value=False)
    auto_rate = st.sidebar.slider("Auto-run rate (ticks/sec)", 0.2, 4.0, 1.2, 0.1)
    auto_refresh = st.sidebar.toggle("Auto-refresh UI", value=True)

    st.sidebar.markdown("### Manual Step")
    if st.sidebar.button("Advance +1 Tick"):
        next_tick = full_state.tick + 1
        batch = generate_synthetic_batch(next_tick, full_state.sources, st.session_state.run_config, full_state.injections)
        append_journal("tick_batch", next_tick, {"events": [asdict(e) for e in batch]})
        st.experimental_rerun()

    if st.sidebar.button("Advance +5 Ticks"):
        t = full_state.tick
        for k in range(1, 6):
            next_tick = t + k
            batch = generate_synthetic_batch(next_tick, full_state.sources, st.session_state.run_config, full_state.injections)
            append_journal("tick_batch", next_tick, {"events": [asdict(e) for e in batch]})
        st.experimental_rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("## Replay")
    view_tick = st.sidebar.slider("Replay tick", 0, max(0, full_state.tick), full_state.tick)
    is_live = (view_tick == full_state.tick)

    st.sidebar.markdown("---")
    st.sidebar.markdown("## Proof: Determinism")
    if st.sidebar.button("Replay Consistency Check"):
        s1 = rebuild_state(journal, st.session_state.run_config, until_tick=view_tick)
        s2 = rebuild_state(journal, st.session_state.run_config, until_tick=view_tick)
        h1 = compute_state_hash(s1)
        h2 = compute_state_hash(s2)
        st.session_state.last_consistency = {"ok": (h1 == h2), "h1": h1, "h2": h2, "tick": view_tick, "ts": time.time()}

    cs = st.session_state.get("last_consistency")
    if cs:
        if cs["ok"]:
            st.sidebar.success(f"PASS tick={cs['tick']} hash={cs['h1']}")
        else:
            st.sidebar.error(f"FAIL tick={cs['tick']} h1={cs['h1']} h2={cs['h2']}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("## Injectors (Journaled)")
    # These commit operator_cmd so replay is consistent.
    inj = full_state.injections
    d = st.sidebar.slider("Dropout rate", 0.0, 0.9, float(inj.dropout_rate), 0.05)
    s = st.sidebar.slider("Spoof jitter", 0.0, 1.0, float(inj.spoof_jitter), 0.05)
    c = st.sidebar.slider("Schema corrupt", 0.0, 0.9, float(inj.schema_corrupt_rate), 0.05)
    i = st.sidebar.slider("Incident spike", 0.0, 0.9, float(inj.incident_spike_rate), 0.05)
    storm = st.sidebar.toggle("Storm mode", value=bool(inj.storm_mode))

    if st.sidebar.button("Commit Injectors"):
        append_journal(
            "operator_cmd",
            full_state.tick,
            {"cmd": "set_injections", "dropout_rate": d, "spoof_jitter": s, "schema_corrupt_rate": c, "incident_spike_rate": i, "storm_mode": storm},
        )
        st.experimental_rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("## Visual Modes")
    enable_particles = st.sidebar.toggle("Particles background", value=True)
    explain_mode = st.sidebar.toggle("WHY mode (overlays)", value=True)
    counterfactual_mode = st.sidebar.toggle("Counterfactual shadows", value=True)
    blame_vectors_mode = st.sidebar.toggle("Blame vectors", value=True)
    heading_vectors_mode = st.sidebar.toggle("Heading vectors", value=True)
    show_corridor = st.sidebar.toggle("Show corridor", value=True)
    show_geofence = st.sidebar.toggle("Show geofence", value=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("## Freeze-frame")
    if st.sidebar.button("Explain this moment (freeze)"):
        st.session_state.explain_freeze_tick = view_tick
    if st.sidebar.button("Clear freeze"):
        st.session_state.explain_freeze_tick = None

    return (
        view_tick, is_live,
        enable_particles, explain_mode, counterfactual_mode, blame_vectors_mode, heading_vectors_mode,
        show_corridor, show_geofence,
        auto_run, auto_rate, auto_refresh
    )

# ================================================================
# UI Contract Export (frontend-ready payload)
# ================================================================

def export_ui_contract(state: SystemState) -> Dict[str, Any]:
    integrity = _global_integrity_score(state)
    threat = compute_threat_level(state)

    def latest(sid: str) -> Optional[EventEnvelope]:
        evs = state.events.get(sid, [])
        return evs[-1] if evs else None

    entities = []
    for sid in ["ship", "air"]:
        ev = latest(sid)
        if not ev:
            continue
        src = state.sources.get(sid)
        trust = state.trust_history.get(sid, [])[-1].total if state.trust_history.get(sid) else 1.0
        entities.append({
            "id": sid,
            "type": sid,
            "lat": float(ev.payload.get("lat", 0.0)),
            "lon": float(ev.payload.get("lon", 0.0)),
            "heading": float(ev.payload.get("heading", 0.0)),
            "speed": float(ev.payload.get("speed", 0.0)),
            "alt": float(ev.payload.get("alt", 0.0)) if sid == "air" else 0.0,
            "trust": float(trust),
            "manual": bool(src.manual_mode) if src else False,
            "enabled": bool(src.enabled) if src else True,
            "tick": int(state.tick),
        })

    alerts = []
    for ins in state.insights[-120:]:
        if ins.level != "Critical":
            continue
        lat, lon = 40.0, -80.0
        if ins.source_id in ("ship", "air"):
            ev = latest(ins.source_id)
            if ev:
                lat, lon = float(ev.payload.get("lat", lat)), float(ev.payload.get("lon", lon))
        alerts.append({"level": ins.level, "kind": ins.kind, "source": ins.source_id, "lat": lat, "lon": lon, "tick": ins.tick})

    return {
        "tick": int(state.tick),
        "integrity": float(integrity),
        "integrity_label": integrity_state_label(integrity),
        "threat": threat.value,
        "entities": entities,
        "alerts": alerts,
        "injections": asdict(state.injections),
    }

# ================================================================
# Map Semantics Helpers
# ================================================================

def trust_color_rgba(trust: float, manual: bool) -> List[int]:
    if manual:
        return [255, 202, 40, 215]  # amber
    if trust < 0.4:
        return [255, 59, 92, 220]   # red
    if trust < 0.6:
        return [255, 138, 61, 200]  # orange
    return [0, 224, 255, 205]       # cyan

def threat_ring_rgba(threat: ThreatLevel) -> List[int]:
    if threat == ThreatLevel.GREEN:
        return [60, 255, 152, 80]
    if threat == ThreatLevel.AMBER:
        return [255, 202, 40, 85]
    if threat == ThreatLevel.ORANGE:
        return [255, 138, 61, 95]
    if threat == ThreatLevel.RED:
        return [255, 59, 92, 110]
    return [167, 183, 196, 80]

def deg2rad(d: float) -> float:
    return d * math.pi / 180.0

def heading_endpoint(lat: float, lon: float, heading_deg: float, magnitude_km: float) -> Tuple[float, float]:
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * math.cos(deg2rad(lat))
    h = deg2rad(heading_deg)
    d_north = math.cos(h) * magnitude_km
    d_east = math.sin(h) * magnitude_km
    dlat = d_north / km_per_deg_lat
    dlon = d_east / max(1e-6, km_per_deg_lon)
    return (lat + dlat, lon + dlon)

def build_trails(state: SystemState, window: int = 60) -> List[Dict[str, Any]]:
    out = []
    for sid in ["ship", "air"]:
        evs = state.events.get(sid, [])
        if len(evs) < 2:
            continue
        tail = evs[-window:]
        path = [[float(e.payload.get("lon", 0.0)), float(e.payload.get("lat", 0.0))] for e in tail if isinstance(e.payload, dict)]
        trust = state.trust_history.get(sid, [])[-1].total if state.trust_history.get(sid) else 1.0
        manual = state.sources.get(sid).manual_mode if sid in state.sources else False
        out.append({"id": sid, "path": path, "color": trust_color_rgba(float(trust), bool(manual))})
    return out

def build_heading_vectors(ui: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for e in ui["entities"]:
        lat, lon = float(e["lat"]), float(e["lon"])
        heading = float(e.get("heading", 0.0))
        speed = float(e.get("speed", 0.0))
        trust = float(e.get("trust", 1.0))
        manual = bool(e.get("manual", False))

        base = 18.0 if e["type"] == "air" else 6.5
        magnitude_km = base * float(np.clip(speed / (500.0 if e["type"] == "air" else 30.0), 0.2, 1.0))

        lat2, lon2 = heading_endpoint(lat, lon, heading, magnitude_km)
        out.append({
            "type": e["type"],
            "trust": trust,
            "manual": manual,
            "start": [lon, lat],
            "end": [lon2, lat2],
            "color": trust_color_rgba(trust, manual),
        })
    return out

def build_blame_vectors(state: SystemState, ui: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for ent in ui["entities"]:
        sid = ent["id"]
        hist = state.trust_history.get(sid, [])
        if len(hist) < 2 or hist[-1].deltas is None:
            continue
        d = hist[-1].deltas
        lat, lon = float(ent["lat"]), float(ent["lon"])

        components = [
            ("freshness", float(d.freshness)),
            ("schema", float(d.schema)),
            ("dropout", float(d.dropout)),
            ("drift", -float(d.drift)),
        ]
        directions = [0, 90, 180, 270]
        for (name, dv), hdg in zip(components, directions):
            if abs(dv) < 0.08:
                continue
            mag = float(np.clip(abs(dv) * 40.0, 2.0, 11.0))
            lat2, lon2 = heading_endpoint(lat, lon, float(hdg), mag)
            bad = (dv < 0)
            color = [255, 59, 92, 210] if bad else [0, 224, 255, 170]
            out.append({"component": name, "delta": dv, "start": [lon, lat], "end": [lon2, lat2], "color": color, "sid": sid})
    return out

def build_counterfactuals(state: SystemState, ui: Dict[str, Any]) -> List[Dict[str, Any]]:
    pending = pending_recommendations(state)
    if not pending:
        return []
    pos = {e["id"]: (float(e["lat"]), float(e["lon"])) for e in ui["entities"]}
    out = []
    for rec in pending[-12:]:
        sid = rec.target_source_id
        lat, lon = pos.get(sid, (40.0, -80.0))
        base_mag = 9.0 if rec.level == "Critical" else 6.0
        lat_ex, lon_ex = heading_endpoint(lat, lon, 45.0, base_mag)
        lat_rj, lon_rj = heading_endpoint(lat, lon, 225.0, base_mag)
        out.append({"mode": "IF_EXECUTED", "rec_id": rec.rec_id[:10], "action": rec.action_class, "level": rec.level, "start": [lon, lat], "end": [lon_ex, lat_ex], "color": [0, 224, 255, 120]})
        out.append({"mode": "IF_REJECTED", "rec_id": rec.rec_id[:10], "action": rec.action_class, "level": rec.level, "start": [lon, lat], "end": [lon_rj, lat_rj], "color": [255, 202, 40, 110]})
    return out

# ================================================================
# Map Renderer (PyDeck / deck.gl)
# ================================================================

def render_ops_map(
    state: SystemState,
    explain_mode: bool,
    counterfactual_mode: bool,
    blame_vectors_mode: bool,
    heading_vectors_mode: bool,
    show_corridor: bool,
    show_geofence: bool,
):
    ui = export_ui_contract(state)
    entities = ui["entities"]
    alerts = ui["alerts"]
    trails = build_trails(state, window=70)
    threat = ThreatLevel(ui["threat"]) if "threat" in ui else compute_threat_level(state)

    for e in entities:
        e["color"] = trust_color_rgba(float(e.get("trust", 1.0)), bool(e.get("manual", False)))

    if entities:
        center_lat = float(np.mean([e["lat"] for e in entities]))
        center_lon = float(np.mean([e["lon"] for e in entities]))
    else:
        center_lat, center_lon = 40.0, -80.0

    layers: List[pdk.Layer] = []

    # Corridor polygon
    if show_corridor:
        corridor = [{
            "poly": [[-82, 38], [-82, 42], [-78, 42], [-78, 38]],
            "fill": threat_ring_rgba(threat),
        }]
        layers.append(
            pdk.Layer(
                "PolygonLayer",
                data=corridor,
                get_polygon="poly",
                get_fill_color="fill",
                get_line_color=[255, 255, 255, 35],
                line_width_min_pixels=1,
                stroked=True,
                filled=True,
                pickable=False,
            )
        )

    # Geofence circle (around center)
    if show_geofence:
        geofence = [{"lon": center_lon, "lat": center_lat}]
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=geofence,
                get_position=["lon", "lat"],
                get_radius=140000,
                radius_min_pixels=10,
                radius_max_pixels=120,
                get_fill_color=[0, 0, 0, 0],
                get_line_color=threat_ring_rgba(threat),
                line_width_min_pixels=2,
                stroked=True,
                filled=False,
                pickable=False,
            )
        )

    # Trails
    if trails:
        layers.append(
            pdk.Layer(
                "PathLayer",
                data=trails,
                get_path="path",
                get_color="color",
                width_min_pixels=2,
                width_max_pixels=6,
                opacity=0.60,
                pickable=False,
            )
        )

    # Critical alert rings
    if alerts:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=alerts,
                get_position=["lon", "lat"],
                get_radius=70000,
                radius_min_pixels=6,
                radius_max_pixels=90,
                get_fill_color=[255, 59, 92, 25],
                get_line_color=[255, 59, 92, 170],
                line_width_min_pixels=2,
                stroked=True,
                filled=False,
                pickable=True,
            )
        )

    # Heading vectors
    if heading_vectors_mode and entities:
        hv = build_heading_vectors(ui)
        layers.append(
            pdk.Layer(
                "LineLayer",
                data=hv,
                get_source_position="start",
                get_target_position="end",
                get_color="color",
                width_min_pixels=2,
                width_max_pixels=6,
                opacity=0.78,
                pickable=True,
            )
        )

    # Entities
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=entities,
            get_position=["lon", "lat"],
            get_radius=16000,
            radius_min_pixels=6,
            radius_max_pixels=20,
            get_fill_color="color",
            get_line_color=[255, 255, 255, 50],
            line_width_min_pixels=1,
            pickable=True,
            auto_highlight=True,
        )
    )

    # WHY overlays
    if explain_mode:
        if blame_vectors_mode and entities:
            bv = build_blame_vectors(state, ui)
            if bv:
                layers.append(
                    pdk.Layer(
                        "LineLayer",
                        data=bv,
                        get_source_position="start",
                        get_target_position="end",
                        get_color="color",
                        width_min_pixels=2,
                        width_max_pixels=6,
                        opacity=0.66,
                        pickable=True,
                    )
                )

        if counterfactual_mode:
            cf = build_counterfactuals(state, ui)
            if cf:
                layers.append(
                    pdk.Layer(
                        "LineLayer",
                        data=cf,
                        get_source_position="start",
                        get_target_position="end",
                        get_color="color",
                        width_min_pixels=2,
                        width_max_pixels=4,
                        opacity=0.55,
                        pickable=True,
                    )
                )

    view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=5.2, pitch=55, bearing=-20)

    tooltip = {
        "html": """
        <div class='ops-mono'>
          <div><b>{type}</b> · trust={trust} · manual={manual}</div>
          <div>tick={tick} · lat={lat} lon={lon}</div>
          <div style='opacity:0.9'>heading={heading} speed={speed}</div>
        </div>
        """,
        "style": {
            "backgroundColor": "rgba(6,9,12,0.94)",
            "color": "#e7f0f6",
            "border": "1px solid rgba(255,255,255,0.10)",
        },
    }

    # Map style: mapbox if key exists, else carto (no key needed)
    mapbox_key = os.getenv("MAPBOX_API_KEY", "")
    map_style = "mapbox://styles/mapbox/dark-v11" if mapbox_key else "carto-darkmatter"

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style=map_style,
        tooltip=tooltip,
    )

    st.markdown("<div class='ops-card'><div class='ops-label'>Tactical Ops Map</div>", unsafe_allow_html=True)
    st.pydeck_chart(deck, use_container_width=True, height=560)
    st.markdown("</div>", unsafe_allow_html=True)

# ================================================================
# Panels
# ================================================================

def render_threat_doctrine_panel(state: SystemState):
    threat = compute_threat_level(state)
    cls = THREAT_COLOR_CLASS[threat]
    st.markdown("<div class='ops-card'><div class='ops-label'>Threat Doctrine</div>", unsafe_allow_html=True)
    st.markdown(f"Current: <span class='pill {cls}'>{threat.value}</span>", unsafe_allow_html=True)

    rows = []
    for r in THREAT_DOCTRINE:
        rows.append({
            "Level": r["level"].value,
            "Rule": r["rule"],
            "Meaning": r["meaning"],
            "Ops Posture": r["ops"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=220)
    st.markdown("</div>", unsafe_allow_html=True)

def render_source_controls(state: SystemState, live_tick: int):
    st.markdown("<div class='ops-card'><div class='ops-label'>Feeds · Manual Overrides · Operator Weights</div>", unsafe_allow_html=True)
    cols = st.columns(len(state.sources))
    staged: Dict[str, Dict[str, Any]] = {}

    for col, (sid, src) in zip(cols, state.sources.items()):
        with col:
            st.markdown(f"**{src.name}**")
            enabled = st.checkbox("Enabled", value=src.enabled, key=f"enabled-{sid}")
            manual = st.checkbox("Manual", value=src.manual_mode, key=f"manual-{sid}")
            weight = st.slider("Operator weight", 0.0, 1.0, float(src.operator_weight), 0.05, key=f"weight-{sid}")
            staged[sid] = {"enabled": enabled, "manual_mode": manual, "operator_weight": weight}

    if st.button("Commit Source Settings"):
        for sid, desired in staged.items():
            cur = state.sources[sid]
            if (desired["enabled"] != cur.enabled) or (desired["manual_mode"] != cur.manual_mode) or (abs(desired["operator_weight"] - cur.operator_weight) > 1e-9):
                append_journal("operator_cmd", live_tick, {"cmd": "set_source", "source_id": sid, **desired})
        st.experimental_rerun()

    st.caption("All changes emit operator_cmd into the tamper-evident journal.")
    st.markdown("</div>", unsafe_allow_html=True)

def render_entity_details(state: SystemState):
    st.markdown("<div class='ops-card'><div class='ops-label'>Entity Detail</div>", unsafe_allow_html=True)
    choices = ["ship", "air"]
    sel = st.selectbox("Select entity", choices, index=0)
    evs = state.events.get(sel, [])
    src = state.sources.get(sel)
    trust_hist = state.trust_history.get(sel, [])

    if not evs:
        st.info("No telemetry yet for this entity.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    last = evs[-1].payload
    t = trust_hist[-1] if trust_hist else None

    c1, c2, c3 = st.columns(3)
    c1.metric("Enabled", str(bool(src.enabled) if src else True))
    c2.metric("Manual", str(bool(src.manual_mode) if src else False))
    c3.metric("Operator weight", f"{float(src.operator_weight) if src else 1.0:.2f}")

    st.markdown("**Latest payload**")
    st.json(last)

    if t:
        st.markdown("**Trust (explainable)**")
        st.write(f"Total: `{t.total:.2f}`  · Base: `{t.base:.2f}`")
        st.json(asdict(t.components))
        if t.blame:
            st.markdown("**Blame**")
            for b in t.blame[:6]:
                st.write(f"- {b}")

    st.markdown("</div>", unsafe_allow_html=True)

def render_insights(state: SystemState):
    st.markdown("<div class='ops-card'><div class='ops-label'>Insight Ladder</div>", unsafe_allow_html=True)
    if not state.insights:
        st.info("No insights yet.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    filt = st.text_input("Filter insights (substring)", value="", placeholder="e.g. trust, schema, threat, ship…")
    items = state.insights[-140:]
    if filt.strip():
        f = filt.strip().lower()
        items = [i for i in items if f in (i.msg + i.kind + i.source_id + i.level).lower()]

    for ins in reversed(items[-18:]):
        header = f"{ins.level} · {ins.kind.upper()} · {ins.source_id.upper()} · t={ins.tick} — {ins.msg}"
        with st.expander(header, expanded=(ins.level == "Critical")):
            if ins.level == "Critical":
                st.markdown("<span class='critical-dot'></span> <b>Critical</b>", unsafe_allow_html=True)
            st.json(ins.metrics)

    st.markdown("</div>", unsafe_allow_html=True)

def render_recommendations(state: SystemState, is_live: bool):
    st.markdown("<div class='ops-card'><div class='ops-label'>Recommendations · Human Gate</div>", unsafe_allow_html=True)
    pending = pending_recommendations(state)
    if not pending:
        st.info("No pending recommendations.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    for rec in reversed(pending[-12:]):
        gate = "Approval required" if rec.requires_approval else "Auto-eligible"
        header = f"{rec.level} · {rec.action} · {gate} · conf={rec.confidence:.2f} · threat={rec.threat_at_issue}"
        with st.expander(header):
            if rec.requires_approval:
                st.warning(rec.blocked_reason or "Requires approval.")
            st.write(rec.rationale)
            st.caption(f"Integrity@issue={rec.integrity_at_issue:.2f} · Load@issue={rec.operator_load_at_issue} · Threat@issue={rec.threat_at_issue}")

            if not is_live:
                st.info("Replay mode: decisions disabled. Move replay tick to latest to act.")
                continue

            comment = st.text_input(f"Comment for {rec.rec_id}", key=f"c-{rec.rec_id}", placeholder="Optional operator note…")
            c1, c2, c3 = st.columns(3)
            if c1.button("Approve", key=f"a-{rec.rec_id}"):
                append_journal("decision", state.tick, {"rec_id": rec.rec_id, "choice": "approve", "comment": comment})
                st.experimental_rerun()
            if c2.button("Reject", key=f"r-{rec.rec_id}"):
                append_journal("decision", state.tick, {"rec_id": rec.rec_id, "choice": "reject", "comment": comment})
                st.experimental_rerun()
            if c3.button("Defer", key=f"d-{rec.rec_id}"):
                append_journal("decision", state.tick, {"rec_id": rec.rec_id, "choice": "defer", "comment": comment})
                st.experimental_rerun()

    st.markdown("</div>", unsafe_allow_html=True)

def render_quarantine(state: SystemState):
    st.markdown("<div class='ops-card'><div class='ops-label'>Schema Quarantine Lane</div>", unsafe_allow_html=True)
    if not state.quarantine:
        st.info("No rejected/quarantined events yet.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    rows = []
    for q in state.quarantine[-30:]:
        e = q.envelope
        rows.append({
            "Tick": e.ingest_tick,
            "Source": e.source_id,
            "Disposition": q.disposition.value,
            "Schema": e.schema_version,
            "EventTS": f"{e.event_ts:.2f}",
            "Reason": q.reason,
        })
    st.dataframe(pd.DataFrame(rows), height=280, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

def render_audit(journal: List[Dict[str, Any]], full_state: SystemState, view_tick: int):
    st.markdown("<div class='ops-card'><div class='ops-label'>Audit · Journal Tail · Capsules</div>", unsafe_allow_html=True)
    tail = journal[-35:]
    rows = []
    for e in tail:
        rows.append({
            "seq": e["seq"],
            "tick": e["tick"],
            "kind": e["kind"],
            "hash": str(e.get("hash", ""))[:12],
            "payload_preview": canonical_json(e["payload"])[:140],
        })
    st.dataframe(pd.DataFrame(rows), height=280, use_container_width=True)

    replay_capsule = {
        "config_hash": st.session_state.config_hash,
        "config": st.session_state.run_config,
        "journal": journal,
        "final_state_hash": compute_state_hash(full_state),
    }
    st.download_button(
        "Download replay_capsule.json",
        data=canonical_json(replay_capsule).encode("utf-8"),
        file_name="replay_capsule.json",
        mime="application/json",
    )

    ui_payload = export_ui_contract(full_state)
    st.download_button(
        "Download ui_state.json (frontend contract)",
        data=canonical_json(ui_payload).encode("utf-8"),
        file_name="ui_state.json",
        mime="application/json",
    )

    st.caption(f"Viewing tick: {view_tick} / latest: {full_state.tick}")
    st.markdown("</div>", unsafe_allow_html=True)

def render_freeze_frame_report(journal: List[Dict[str, Any]], freeze_tick: int):
    st.markdown("<div class='ops-card'><div class='ops-label'>Explain This Moment · Freeze-Frame</div>", unsafe_allow_html=True)
    s = rebuild_state(journal, st.session_state.run_config, until_tick=freeze_tick)
    ui = export_ui_contract(s)

    st.markdown(f"**Freeze tick:** `{freeze_tick}` · **Threat:** `{ui['threat']}` · **Integrity:** `{ui['integrity']:.2f}` · **State:** `{ui['integrity_label']}`")

    nearby = [i for i in s.insights if (freeze_tick - 10) <= i.tick <= freeze_tick]
    crits = [i for i in nearby if i.level == "Critical"]
    warns = [i for i in nearby if i.level == "Warning"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Nearby Insights", str(len(nearby)))
    c2.metric("Critical", str(len(crits)))
    c3.metric("Warning", str(len(warns)))

    if crits:
        st.markdown("**Critical chain (most recent first):**")
        for i in reversed(crits[-10:]):
            st.write(f"- `{i.tick}` · `{i.kind}` · `{i.source_id}` · {i.msg}")

    pending = pending_recommendations(s)
    st.markdown("---")
    st.markdown(f"**Pending recommendations at tick {freeze_tick}:** `{len(pending)}`")
    if pending:
        for r in pending[-10:]:
            st.write(f"- `{r.level}` · `{r.action_class}` · target=`{r.target_source_id}` · gate=`{r.requires_approval}` · conf=`{r.confidence:.2f}` · threat=`{r.threat_at_issue}`")

    head = st.session_state.journal[-1].get("hash", "")[:12] if st.session_state.journal else "—"
    st.markdown("---")
    st.markdown("**Audit watermark:**")
    st.code(
        "\n".join([
            f"CONFIG_HASH   : {st.session_state.config_hash[:16]}",
            f"JOURNAL_HEAD  : {head}",
            f"STATE_HASH@T  : {compute_state_hash(s)}",
            f"FREEZE_TICK   : {freeze_tick}",
        ]),
        language="text",
    )

    st.markdown("</div>", unsafe_allow_html=True)

# ================================================================
# Main
# ================================================================

def main():
    init_run_state()

    run_cfg = st.session_state.run_config
    policy = PolicyConfig(**run_cfg.get("policy", {}))
    journal = st.session_state.journal

    # Hard validation before any rebuild
    ok, msg = validate_journal(journal)
    if not ok:
        st.error(f"Journal validation failed: {msg}")
        st.stop()
    ok2, msg2 = validate_journal_semantics(journal)
    if not ok2:
        st.error(f"Journal semantic validation failed: {msg2}")
        st.stop()

    full_state = rebuild_state(journal, run_cfg, until_tick=None)

    (
        view_tick, is_live,
        enable_particles, explain_mode, counterfactual_mode, blame_vectors_mode, heading_vectors_mode,
        show_corridor, show_geofence,
        auto_run, auto_rate, auto_refresh
    ) = render_sidebar_controls(full_state, journal)

    render_particles_background(enable_particles)

    # Optional live loop (safe-ish for demo)
    if auto_run:
        # Advance one tick per loop based on rate; rerun continuously.
        # IMPORTANT: auto actions are still governed (doctrine + policy).
        next_tick = full_state.tick + 1
        batch = generate_synthetic_batch(next_tick, full_state.sources, run_cfg, full_state.injections)
        append_journal("tick_batch", next_tick, {"events": [asdict(e) for e in batch]})

        # throttle
        sleep_s = float(max(0.05, 1.0 / float(auto_rate)))
        time.sleep(sleep_s)
        if auto_refresh:
            st.experimental_rerun()

    view_state = rebuild_state(journal, run_cfg, until_tick=view_tick)

    render_header()
    render_statusbar(view_state, policy, journal)

    freeze_tick = st.session_state.get("explain_freeze_tick")
    if freeze_tick is not None:
        render_freeze_frame_report(journal, int(freeze_tick))

    left, right = st.columns([1.60, 1.0])
    with left:
        render_ops_map(
            view_state,
            explain_mode=explain_mode,
            counterfactual_mode=counterfactual_mode,
            blame_vectors_mode=blame_vectors_mode,
            heading_vectors_mode=heading_vectors_mode,
            show_corridor=show_corridor,
            show_geofence=show_geofence,
        )
        render_quarantine(view_state)

    with right:
        render_threat_doctrine_panel(view_state)
        render_source_controls(full_state, live_tick=full_state.tick)
        render_entity_details(view_state)
        render_insights(view_state)
        render_recommendations(full_state if is_live else view_state, is_live=is_live)

    st.markdown("---")
    render_audit(journal, full_state, view_tick)

    st.markdown("<div class='ops-card'><div class='ops-label'>Invariants</div>", unsafe_allow_html=True)
    try:
        integ = _global_integrity_score(full_state)
        assert 0.0 <= integ <= 1.0
        for sid, stats in full_state.stats.items():
            assert stats.accepted + stats.rejected + stats.quarantined == stats.attempts
            if stats.watermark_ts is not None:
                assert not math.isnan(float(stats.watermark_ts))
        ok, _ = validate_journal(journal)
        ok2, _ = validate_journal_semantics(journal)
        assert ok and ok2
        st.success("All invariants passed (journal integrity + semantic monotonicity + bounded scores + stats accounting).")
    except Exception as e:
        st.error(f"Invariant violated: {type(e).__name__}: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
