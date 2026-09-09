
import os
import re

FRONTAL_VIEWS = {"AP", "PA", "APPA", "PA LLD", "AP LLD", "AP RLD", "PA RLD"}
LATERAL_VIEWS = {"LATERAL", "LL", "XTABLE LATERAL"}

def _as_str(x):
    if x is None:
        return ""
    if isinstance(x, float) and str(x).lower() == "nan":
        return ""
    return str(x).strip()

def normalize_view(v):
    v = _as_str(v).upper()
    if not v or v in {"NAN", "NONE", "NULL"}:
        return "unk"
    return v

def view_group(v):
    v = normalize_view(v)
    if v in FRONTAL_VIEWS:
        return "frontal"
    if v in LATERAL_VIEWS:
        return "lateral"
    return "other"

def first_item(x, default=""):
    if isinstance(x, (list, tuple)):
        return x[0] if len(x) > 0 else default
    return x if x is not None else default

def serialize_phrases(value, sep_token="[SEP]"):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        parts = []
        for x in value:
            x = _as_str(x)
            if x:
                parts.append(x)
        return (" %s " % sep_token).join(parts).strip()
    return _as_str(value)

def get_current_image_path(example):
    if "anchor_scan" in example and isinstance(example["anchor_scan"], dict):
        return first_item(example["anchor_scan"].get("image_path"), "")
    return first_item(example.get("image_path"), "")

def get_current_view(example):
    if "anchor_scan" in example and isinstance(example["anchor_scan"], dict):
        return normalize_view(first_item(example["anchor_scan"].get("view_position"), "unk"))
    return normalize_view(example.get("viewposition", example.get("view_position", "unk")))

def get_prior_image_path(example):
    ps = example.get("prior_study")
    if isinstance(ps, dict):
        latest = ps.get("latest_study")
        if isinstance(latest, dict):
            p = latest.get("image_path")
            p = first_item(p, "")
            if p:
                return p
    return first_item(example.get("prior_image_path"), "")

def get_prior_view(example):
    ps = example.get("prior_study")
    if isinstance(ps, dict):
        latest = ps.get("latest_study")
        if isinstance(latest, dict):
            v = latest.get("view_position", latest.get("viewposition", "unk"))
            return normalize_view(first_item(v, "unk"))
    return normalize_view(example.get("prior_viewposition", example.get("prior_view_position", "unk")))

def get_target_report(example, target_report_field="auto"):
    if target_report_field and target_report_field != "auto":
        val = example.get(target_report_field)
        if val:
            return val
    for key in ["findings", "reports", "report"]:
        val = example.get(key)
        if val:
            return val
    # avoid using structured target unless nothing else exists
    val = serialize_phrases(example.get("findings_factual_serialization") or example.get("reports_pure"))
    return val if val else "no findings ."

def get_indication_text(example):
    ind = example.get("indication_pure") or example.get("indication") or ""
    ind = _as_str(ind)
    if ind:
        return "[INDICATION] " + ind
    return "[NHI]"

def get_structured_prior_report_text(example, split, use_pr_in_eval=False):
    # MLRG-style leakage-safe setting: validation/test previous report is unavailable.
    if split in ("val", "test") and not use_pr_in_eval:
        return "[NHPR]"

    # 1) nested PriorRG/MLRG format
    ps = example.get("prior_study")
    if isinstance(ps, dict):
        latest = ps.get("latest_study")
        if isinstance(latest, dict):
            structured = serialize_phrases(latest.get("findings_factual_serialization"))
            if structured:
                return "[PRIOR_REPORT] " + structured

    # 2) possible flat structured fields
    for key in [
        "prereport_factual_serialization",
        "prereport_pure",
        "prior_report_factual_serialization",
        "prior_report_pure",
        "previous_report_factual_serialization",
        "previous_report_pure",
    ]:
        structured = serialize_phrases(example.get(key))
        if structured:
            return "[PRIOR_REPORT] " + structured

    # Do NOT fall back to raw prereport in this leakage-safe structured-prior setup.
    return "[NHPR]"
