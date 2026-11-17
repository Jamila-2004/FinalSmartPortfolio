"""
Grasshopper predictive custom node  (Rhino 8 — GhPython "Python 3 / CPython" component).

Loads the trained pipeline (models/model.pkl) and predicts the architectural
targets + healing score from a patient profile in real time.

--- Setup in Grasshopper (Rhino 8) ---
1. Drop a "Python 3 Script" component on the canvas.
2. Add inputs (item access) matching the names in PROFILE_INPUTS below:
       x1_depression, x2_anxiety, x3_stress, x4_trauma, x5_substance_sev,
       x5_substance_type, x6_sensory, x6b_spatial_trigger, x7_trigger,
       x8_behavior, x9_privacy, x10_stage, x11_age, x12_gender, x15_admission
   plus a string input  model_path  pointing to models/model.pkl
3. Add outputs:  Y_score  and  targets (a tree/dict of Y1..Y15).
4. The component needs scikit-learn + joblib in the Rhino 8 python env:
       (Rhino 8) ScriptEditor > ... > pip install scikit-learn joblib pandas

This script also runs standalone for testing:  python gh/predict_node.py
"""
import os
import sys

PROFILE_INPUTS = [
    "x1_depression", "x2_anxiety", "x3_stress", "x4_trauma", "x5_substance_sev",
    "x5_substance_type", "x6_sensory", "x6b_spatial_trigger", "x7_trigger",
    "x8_behavior", "x9_privacy", "x10_stage", "x11_age", "x12_gender", "x15_admission",
]

# Defaults for any input not yet wired (so the node never crashes mid-setup) + numeric coercion
# (GH panels can hand in text, so numbers are forced to float).
_DEFAULTS = {
    "x1_depression": 20.0, "x2_anxiety": 20.0, "x3_stress": 0.5, "x4_trauma": 30.0,
    "x5_substance_sev": 0.5, "x5_substance_type": "tramadol", "x6_sensory": 0.5,
    "x6b_spatial_trigger": "glare", "x7_trigger": "family-conflict",
    "x8_behavior": "withdrawal", "x9_privacy": 0.5, "x10_stage": "detox-1",
    "x11_age": 32.0, "x12_gender": 1.0, "x15_admission": "Madah-10",
}
_NUMERIC_INPUTS = {"x1_depression", "x2_anxiety", "x3_stress", "x4_trauma",
                   "x5_substance_sev", "x6_sensory", "x9_privacy", "x11_age", "x12_gender"}


def _coerce_input(key, value):
    """Fill an unwired input with its default; coerce numeric inputs to float."""
    if value is None or value == "":
        value = _DEFAULTS[key]
    if key in _NUMERIC_INPUTS:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(_DEFAULTS[key])
    return str(value)

_CACHE = {}


def _load(model_path):
    if "model" not in _CACHE:
        import joblib, json
        _CACHE["model"] = joblib.load(model_path)
        meta_path = os.path.join(os.path.dirname(model_path), "model_meta.json")
        with open(meta_path) as f:
            _CACHE["meta"] = json.load(f)
    return _CACHE["model"], _CACHE["meta"]


def predict(profile, model_path, orientation=None):
    """profile: dict of the PROFILE_INPUTS. Returns (healing_score, targets_dict, cluster).

    orientation: optional independent ENVIRONMENT input ("NW" / "NE"). If given it overrides
    the orientation the patient's phase would otherwise get — so you can evaluate the same
    patient's design in NW vs NE. If None, orientation is derived from phase + sensitivity.
    """
    # Self-defending: fill any unwired/None input with its default and coerce numbers to float,
    # so the node never crashes on a missing slider regardless of how it's called.
    profile = {k: _coerce_input(k, (profile or {}).get(k)) for k in PROFILE_INPUTS}

    # Locate src/mapping.py robustly. Rhino's Python 3 component has no real __file__, so also
    # derive the project root from model_path, and fall back to the known project location.
    _src_dirs = []
    if model_path:
        try:
            _root = os.path.dirname(os.path.dirname(os.path.abspath(model_path)))
            _src_dirs.append(os.path.join(_root, "src"))
        except Exception:
            pass
    try:
        _here = os.path.dirname(os.path.abspath(__file__))
        _src_dirs.append(os.path.join(os.path.dirname(_here), "src"))
    except Exception:
        pass
    _src_dirs.append(r"C:\ccas2.6\MainProject\src")  # known project location (last resort)
    for _d in _src_dirs:
        if os.path.isdir(_d) and _d not in sys.path:
            sys.path.insert(0, _d)

    # derive cluster + orientation the same way the dataset was built
    from mapping import assign_cluster, CLUSTERS, recommend_targets, healing_score
    cluster = assign_cluster(profile["x10_stage"], float(profile["x6_sensory"]),
                             float(profile["x4_trauma"]))
    profile = dict(profile)
    profile["x13_cluster"] = cluster
    profile["x14_orientation"] = CLUSTERS[cluster]["orientation"]
    # Independent environment override: force NW/NE and match a same-phase cluster if one exists.
    if orientation in ("NW", "NE"):
        profile["x14_orientation"] = orientation
        for _c, _m in CLUSTERS.items():
            if _m["phase"] == profile["x10_stage"] and _m["orientation"] == orientation:
                profile["x13_cluster"] = _c
                break

    try:
        # Primary path: the trained ML model (.pkl) predicts every recommendation + the score.
        import pandas as pd
        model, meta = _load(model_path)
        row = pd.DataFrame([{k: profile[k] for k in meta["features"]}])
        pred = model.predict(row)[0]
        out = dict(zip(meta["targets"], [round(float(v), 3) for v in pred]))
        score = out.pop("Y_score")
        source = "ml-model (.pkl)"
    except Exception:
        # Fallback (e.g. Rhino's Python can't load the pickle): the same evidence rules the
        # model was trained on — pure Python, no sklearn/joblib needed. Output is equivalent.
        Y = recommend_targets(profile)
        out = {k: round(float(v), 3) for k, v in Y.items()}
        score = round(float(healing_score(profile, Y)), 1)
        source = "rule-engine (fallback)"
    return score, out, profile["x13_cluster"], source


# --- Grasshopper execution: inputs arrive as global vars; outputs are assigned ---
# Detect Rhino by importing RhinoCommon (reliable in the Rhino 8 Python 3 component).
def _in_rhino():
    try:
        import Rhino  # noqa: F401 — only importable inside Rhino / Grasshopper
        return True
    except Exception:
        return False


if _in_rhino():  # pragma: no cover
    profile = {k: _coerce_input(k, globals().get(k)) for k in PROFILE_INPUTS}
    mp = globals().get("model_path") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "models", "model.pkl")
    ori = globals().get("orientation")  # optional independent environment input ("NW"/"NE")
    Y_score, targets, cluster, source = predict(profile, mp, orientation=ori)
    # Expose each predicted design recommendation as its own output (add an output param named
    # like the key to read it): Y1_ceiling_h, ... Y16_screen_openness, plus Y_score, cluster, source.
    for _k, _v in targets.items():
        globals()[_k] = _v

elif __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    mp = os.path.join(here, "..", "models", "model.pkl")
    demo = {
        "x1_depression": 30, "x2_anxiety": 33, "x3_stress": 0.7, "x4_trauma": 60,
        "x5_substance_sev": 0.8, "x5_substance_type": "heroin", "x6_sensory": 0.75,
        "x6b_spatial_trigger": "glare", "x7_trigger": "family-conflict",
        "x8_behavior": "agitation", "x9_privacy": 0.8, "x10_stage": "detox-1",
        "x11_age": 28, "x12_gender": 1, "x15_admission": "Madah-13",
    }
    score, targets, cluster, source = predict(demo, mp)
    print(f"Source: {source}")
    print(f"Assigned cluster: {cluster}")
    print(f"Predicted healing score: {score:.1f}/100")
    print("Design recommendations:")
    for k, v in targets.items():
        print(f"  {k:<22} {v}")
