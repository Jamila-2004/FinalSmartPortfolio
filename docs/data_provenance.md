# Data Provenance — what this dataset is, and what it is not

> **Plain statement:** this is an **evidence-grounded simulation**, *not* a collection of real
> patient records and *not* data extracted from a Grasshopper run. The 500 rows are **simulated
> individuals**. No number is invented arbitrarily — every distribution, parameter and rule traces
> to a cited source or to one of the two expert interviews.

This honesty matters: a graduation jury may ask *"is this real patient data?"*. The correct,
defensible answer is below.

---

## Why simulation (not real records)

500 real psychiatric + addiction records cannot be obtained ethically or legally for an
undergraduate project (patient confidentiality, IRB constraints, the "patient-deception" realities
noted in the Al-Najah interview). Evidence-grounded simulation from validated instruments + expert
interviews is a recognised research method for this exact situation — it lets us study the
**psychology + environment → architecture** relationship without exposing real patients.

---

## The three provenance tiers

| Tier | Columns | How produced | Grounding |
|---|---|---|---|
| **1. Simulated inputs** | X1–X12, X15 (psychology, demographics, admission) | **Sampled** from probability distributions in `src/generate_dataset.py` | Distribution *parameters* from Arabic DASS-21 / PCL-5 / PSS, Egyptian prevalence, UNODC/WHO, and the two interviews |
| **2. Ladybug-validated climate** | **Y3 daylight, Y4 glare, Y5 shading, Y6 ventilation** | Constants are **literature-grounded and validated** by a Grasshopper + Ladybug model under the real El Minya EPW (measured summer direct-sun-hours ≈ 593–605 h; adaptive comfort). **Y4 glare is calibrated** from the measured sun-hours via `data/env_calibration.json`; the rest keep validated literals | climate.onebuilding.org EPW (Minya/Asyut), ASHRAE 55 adaptive comfort |
| **3. Evidence rules** | Y1, Y2, Y7–Y15 + **Y_score** | **Computed** by deterministic evidence-based rules / 8-pillar rubric in `src/mapping.py` | Ulrich supportive design, Terrapin biophilic, trauma-informed design, WELL, the interviews |

Derived columns (X13 cluster, X14 orientation) follow deterministically from recovery stage +
sensitivity. `X16` accommodation is the string echo of target Y14 and is **excluded** from model
features to prevent leakage.

---

## Unit of recommendation = the recovery phase

Recommendations and the healing score are read at the level of the **recovery phase**, not the
individual:

- **Detox-1 (acute):** maximum protection — high privacy/enclosure, low stimulation, strong glare
  and acoustic control, de-escalation proximity, contained volumes, gentle light (sensitive cases
  routed to the NE cluster). Minimal social exposure.
- **Detox-2 (stepping-down):** still protected but beginning to open — more daylight and volume,
  the first "structured choice", easing toward social contact.
- **Rehab-therapy (communal):** outward — high social gradient, belonging/activity spaces (art,
  music, garden), supervised family-interaction spaces, strong biophilia and daylight, openness.

Individual patients/states **populate** each phase and produce the within-phase variation the model
learns. We keep per-patient rows for that variation, but present the program and score per phase.

---

## What the ML model actually does (honest framing)

Because targets are produced by the evidence rules/rubric and the patient features are the inputs to
those rules, the model is effectively **learning to reproduce an evidence-based design mapping** so
it can run in **real time inside Grasshopper** — it is *not* claimed to "discover novel patterns in
real-world outcomes". High R² is therefore expected (the mapping is largely deterministic; injected
design noise sets the ceiling), which is also why a linear model (Ridge) is competitive. The
climate side is *validated* against a real Ladybug simulation under the El Minya EPW — measured
summer direct-sun-hours (≈ 593 h NW / 605 h NE) and adaptive comfort confirm the harsh-solar /
high-ventilation-potential climate, and glare control is calibrated from those measured sun-hours —
which grounds the environmental side outside the rule system.

---

## Stated limitations

- **Simulated population** — not real individuals; external validity is bounded by the cited
  distribution parameters.
- **Arabic PCL-5** is validated in Arabic populations (Algeria/Tunisia/Lebanon) but **not yet
  Egypt-specific** — a documented limitation.
- **Single base-case geometry** — the Ladybug calibration uses one parametric shoebox room per
  orientation, not the full building.
- **Healing score is a design-quality label**, not a measured clinical recovery outcome.
