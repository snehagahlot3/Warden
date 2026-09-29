# -*- coding: utf-8 -*-
"""
Attack-category -> MITRE ATT&CK interpretation layer.

IMPORTANT - SCOPE AND HONESTY CONSTRAINT
=======================================
The primary model output is the *dataset* attack category, i.e. the CIC-IDS2017
`Label` value attached to a network state by the preprocessing pipeline. Those
labels are NOT MITRE ATT&CK tactics and this module does not pretend otherwise.

This file is a *separate, optional* interpretation layer that attaches an
ATT&CK tactic/technique to a dataset category where such a mapping is
defensible from the dataset's own documentation, and yields "Unmapped"
everywhere else. It never changes, overrides or re-labels the model's actual
prediction.

MAPPING EVIDENCE
----------------
Each entry records the CIC-IDS2017 attack as described in the dataset paper
(Sharafaldin et al., ICISSP 2018) and the ATT&CK technique it most closely
resembles in observed network traffic. Confidence is deliberately conservative:

  "high"    - the dataset paper / ATT&CK both describe the same behaviour
  "medium"  - close analogue, behaviour is similar but not identical
  "low"     - weak analogy, reported as suggestive only

Any "- Attempted" variant is a truncated/failed instance of its base attack and
inherits the base attack's mapping, tagged as attempted.
"""

# Base categories first; "- Attempted" / "- Portscan" variants are resolved
# programmatically by `resolve_category` with an explicit override table.
_BASE_CATEGORY_MAP = {
    "BENIGN": {
        "tactic": None,
        "technique_id": None,
        "technique": None,
        "confidence": "n/a",
        "note": "Normal traffic; no ATT&CK mapping is appropriate.",
    },
    "Portscan": {
        "tactic": "Discovery",
        "technique_id": "T1046",
        "technique": "Network Service Discovery",
        "confidence": "high",
        "note": (
            "Port scanning is a direct instance of T1046. CIC-IDS2017 generates "
            "Portscan with nmap; the ATT&CK procedure is the same behaviour."
        ),
    },
    "Infiltration - Portscan": {
        "tactic": "Discovery",
        "technique_id": "T1046",
        "technique": "Network Service Discovery",
        "confidence": "high",
        "note": "Portscan component of the Infiltration attack.",
    },
    "DDoS": {
        "tactic": "Impact",
        "technique_id": "T1498",
        "technique": "Network Denial of Service",
        "confidence": "medium",
        "note": (
            "DDoS is resource exhaustion by traffic volume. The dataset label "
            "does not distinguish volumetric from direct/other DoS subtypes, "
            "so the parent technique T1498 is used rather than a leaf."
        ),
    },
    "DoS Hulk": {
        "tactic": "Impact",
        "technique_id": "T1499",
        "technique": "Endpoint Denial of Service",
        "confidence": "medium",
        "note": (
            "Application-layer HTTP request flood against a single web server; "
            "matches the T1499 application-layer resource exhaustion pattern."
        ),
    },
    "DoS GoldenEye": {
        "tactic": "Impact",
        "technique_id": "T1499",
        "technique": "Endpoint Denial of Service",
        "confidence": "medium",
        "note": "HTTP KeepAlive/HTTP flood against a web server; T1499 pattern.",
    },
    "DoS slowloris": {
        "tactic": "Impact",
        "technique_id": "T1499",
        "technique": "Endpoint Denial of Service",
        "confidence": "medium",
        "note": "Slow HTTP request exhaustion; T1499 application-layer pattern.",
    },
    "DoS Slowhttptest": {
        "tactic": "Impact",
        "technique_id": "T1499",
        "technique": "Endpoint Denial of Service",
        "confidence": "medium",
        "note": "Slow HTTP attack variant from the same tool family as slowloris.",
    },
    "FTP-Patator": {
        "tactic": "Credential Access",
        "technique_id": "T1110.001",
        "technique": "Brute Force: Password Guessing",
        "confidence": "high",
        "note": (
            "Repeated FTP login attempts from one host. CIC labels this a "
            "credential brute force; T1110.001 covers exactly this."
        ),
    },
    "SSH-Patator": {
        "tactic": "Credential Access",
        "technique_id": "T1110.001",
        "technique": "Brute Force: Password Guessing",
        "confidence": "high",
        "note": "Repeated SSH login attempts; T1110.001 password guessing.",
    },
    "Web Attack - Brute Force": {
        "tactic": "Credential Access",
        "technique_id": "T1110.001",
        "technique": "Brute Force: Password Guessing",
        "confidence": "high",
        "note": "Web application login brute force; T1110.001.",
    },
    "Web Attack - Sql Injection": {
        "tactic": "Initial Access",
        "technique_id": "T1190",
        "technique": "Exploit Public-Facing Application",
        "confidence": "medium",
        "note": (
            "SQL injection exploits a public-facing web application. The "
            "dataset label identifies the payload, not the access vector, so "
            "T1190 is the closest parent technique."
        ),
    },
    "Web Attack - XSS": {
        "tactic": "Initial Access",
        "technique_id": "T1189",
        "technique": "Drive-by Compromise",
        "confidence": "low",
        "note": (
            "Cross-site scripting in a web application. XSS is an application "
            "weakness rather than a clean ATT&CK technique; the browser-side "
            "execution is only loosely analogous to T1189. Reported as low "
            "confidence on purpose."
        ),
    },
    "Heartbleed": {
        "tactic": "Credential Access",
        "technique_id": "T1552.004",
        "technique": "Unsecured Credentials: Private Keys",
        "confidence": "medium",
        "note": (
            "Heartbleed (CVE-2014-0160) leaks process memory over TLS, which "
            "can expose keys and credentials. T1552.004 is the closest "
            "credential-exposure technique; the memory disclosure itself is "
            "not directly represented."
        ),
    },
    "Bot": {
        "tactic": "Command and Control",
        "technique_id": "T1071",
        "technique": "Application Layer Protocol",
        "confidence": "low",
        "note": (
            "Botnet command-and-control traffic. The dataset does not reveal "
            "the C2 channel, so only the parent technique is defensible."
        ),
    },
    "Botnet - Attempted": {
        "tactic": "Command and Control",
        "technique_id": "T1071",
        "technique": "Application Layer Protocol",
        "confidence": "low",
        "note": "Failed/truncated botnet C2 attempt.",
    },
    "Infiltration": {
        "tactic": "Initial Access",
        "technique_id": "T1200",
        "technique": "Hardware Additions",
        "confidence": "low",
        "note": (
            "CIC-IDS2017 infiltration drops a downloadable executable onto a "
            "victim machine over Dropbox/HTTPS. That is not T1200 (physical "
            "access); the honest reading is an uncharacterised initial-access "
            "channel, so this entry is explicitly low confidence."
        ),
    },
}

# Variants that must NOT simply inherit their base category.
_VARIANT_OVERRIDES = {
    "Infiltration - Attempted": {
        "tactic": "Initial Access",
        "technique_id": None,
        "technique": None,
        "confidence": "low",
        "note": "Failed infiltration download; no defensible technique mapping.",
    },
}


def resolve_category(category):
    """
    Return the ATT&CK interpretation dict for a dataset attack category.

    Handles the CIC-IDS2017 '- Attempted' suffix by inheriting the base
    category's mapping, and applies explicit overrides where inheritance would
    be wrong. Anything unknown returns an "Unmapped" record.
    """
    category = str(category)

    if category in _BASE_CATEGORY_MAP:
        entry = dict(_BASE_CATEGORY_MAP[category])
    elif category in _VARIANT_OVERRIDES:
        entry = dict(_VARIANT_OVERRIDES[category])
    elif category.endswith(" - Attempted"):
        base = category[: -len(" - Attempted")]
        if base in _BASE_CATEGORY_MAP:
            entry = dict(_BASE_CATEGORY_MAP[base])
            entry["note"] = (
                f"Failed/truncated instance of '{base}'. "
                + entry.get("note", "")
            ).strip()
            entry["confidence"] = _downgrade(entry.get("confidence", "low"))
        else:
            return unmapped(category)
    else:
        return unmapped(category)

    entry["dataset_category"] = category
    return entry


def _downgrade(confidence):
    return {"high": "medium", "medium": "low", "low": "low", "n/a": "n/a"}.get(
        confidence, "low"
    )


def unmapped(category):
    """Explicit record for a category with no defensible ATT&CK analogue."""
    return {
        "dataset_category": str(category),
        "tactic": None,
        "technique_id": None,
        "technique": None,
        "confidence": "unmapped",
        "note": "No defensible MITRE ATT&CK mapping for this dataset category.",
    }


def describe_mapping(category):
    """One-line human-readable summary, e.g. 'Discovery / T1046 (high)'."""
    e = resolve_category(category)
    if e["technique_id"] is None:
        return f"Unmapped ({e['dataset_category']})"
    return f"{e['tactic']} / {e['technique_id']} {e['technique']} ({e['confidence']})"


def full_mapping_table():
    """Every dataset category observed in the processed splits, with mapping."""
    import numpy as np
    from pathlib import Path

    processed = Path(__file__).resolve().parents[2] / "data" / "processed"
    categories = set()
    for split in ("train", "val", "test"):
        cats = np.load(processed / f"{split}_categories.npy", allow_pickle=True)
        categories.update(np.unique(cats).tolist())
    return {c: resolve_category(c) for c in sorted(categories)}
