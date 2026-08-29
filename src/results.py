"""Messergebnisse dauerhaft ablegen.

Eine Zahl ist nur so viel wert wie das, was man ueber ihre Herkunft weiss.
Deshalb haelt jeder Eintrag nicht nur das Ergebnis fest, sondern auch die
Parameter, die Umgebung und einen Fingerabdruck des Codes, der sie erzeugt
hat. Erst damit laesst sich spaeter entscheiden, ob zwei Zahlen ueberhaupt
vergleichbar sind.

Der Anlass war handfest: von den Messungen einer ganzen Arbeitssitzung lag am
Ende nur eine einzige noch auf der Platte, weil jeder Lauf die Logdatei des
vorigen ueberschrieben hatte.
"""
import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results")
SRC_DIR = os.path.dirname(os.path.abspath(__file__))


def code_fingerprint() -> dict:
    """Kurz-Hash je Quelldatei plus ein Gesamthash.

    Ohne Versionsverwaltung ist das die einzige Moeglichkeit festzuhalten,
    welcher Stand eine Zahl erzeugt hat.
    """
    per_file = {}
    combined = hashlib.sha256()
    for name in sorted(os.listdir(SRC_DIR)):
        if not name.endswith(".py"):
            continue
        data = open(os.path.join(SRC_DIR, name), "rb").read()
        digest = hashlib.sha256(data).hexdigest()
        per_file[name] = digest[:12]
        combined.update(name.encode() + digest.encode())
    return {"files": per_file, "combined": combined.hexdigest()[:12]}


def environment() -> dict:
    # Kein Rechnername: fuer die Reproduzierbarkeit zaehlen GPU, torch und
    # Python, nicht wo die Kiste steht.
    env = {"python": platform.python_version()}
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version",
                              "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        env["gpu"] = gpu.stdout.strip().splitlines()[0] if gpu.stdout.strip() else "?"
    except Exception:
        env["gpu"] = "?"
    try:
        import torch
        env["torch"] = torch.__version__
    except Exception:
        pass
    return env


def record(tool: str, params: dict, rows: list[dict], notes: str = "",
           provenance: str = "measured") -> str:
    """Legt einen Messlauf ab und gibt den Pfad zurueck.

    `provenance` unterscheidet frisch gemessene Zahlen von solchen, die
    nachtraeglich aus einem Protokoll uebernommen wurden - das muss man
    spaeter auseinanderhalten koennen.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    payload = {
        "tool": tool,
        "recorded_at": stamp,
        "unix_time": int(time.time()),
        "provenance": provenance,
        "params": params,
        "environment": environment(),
        # Bei nachgetragenen Zahlen waere der heutige Codestand irrefuehrend -
        # er hat sie nicht erzeugt.
        "code": (code_fingerprint() if provenance == "measured"
                 else {"note": "Codestand nicht rekonstruierbar (nachgetragen)"}),
        "notes": notes,
        "rows": rows,
    }
    # Zwei Laeufe in derselben Sekunde duerfen sich nicht ueberschreiben -
    # genau so ging beim ersten Nachtragen ein Eintrag verloren.
    path = os.path.join(RESULTS_DIR, f"{stamp}_{tool}.json")
    suffix = 1
    while os.path.exists(path):
        path = os.path.join(RESULTS_DIR, f"{stamp}_{tool}_{suffix}.json")
        suffix += 1
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path


def load_all() -> list[dict]:
    if not os.path.isdir(RESULTS_DIR):
        return []
    out = []
    for name in sorted(os.listdir(RESULTS_DIR)):
        if name.endswith(".json"):
            with open(os.path.join(RESULTS_DIR, name), encoding="utf-8") as handle:
                entry = json.load(handle)
                entry["_file"] = name
                out.append(entry)
    return out
