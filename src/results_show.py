"""Zeigt die gesicherten Messlaeufe und vergleicht sie.

Ohne Ueberblick nuetzt ein Archiv nichts. `--tool` filtert, `--diff` stellt
zwei Laeufe desselben Werkzeugs nebeneinander - inklusive der Frage, ob
ueberhaupt derselbe Codestand dahintersteckt.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from results import load_all   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", default="", help="nur dieses Werkzeug")
    ap.add_argument("--rows", action="store_true", help="Messzeilen mit ausgeben")
    ap.add_argument("--diff", nargs=2, metavar="N", type=int,
                    help="zwei Laeufe anhand ihrer laufenden Nummer vergleichen")
    args = ap.parse_args()

    entries = [e for e in load_all() if not args.tool or e["tool"] == args.tool]
    if not entries:
        print("Keine Ergebnisse in results/")
        return

    if args.diff:
        return _diff(entries, *args.diff)

    print(f"{'#':<4}{'Zeitpunkt':<18}{'Werkzeug':<18}{'Herkunft':<12}{'Code':<14}Zeilen")
    print("-" * 78)
    for index, e in enumerate(entries):
        code = e["code"].get("combined", "-")
        print(f"{index:<4}{e['recorded_at']:<18}{e['tool']:<18}"
              f"{e['provenance']:<12}{code:<14}{len(e['rows'])}")
        if e.get("notes"):
            print(f"    {e['notes'][:88]}")
        if args.rows:
            for row in e["rows"]:
                print("      " + "  ".join(f"{k}={v}" for k, v in row.items()))
    print(f"\n{len(entries)} Laeufe in results/")


def _diff(entries, a, b):
    try:
        left, right = entries[a], entries[b]
    except IndexError:
        print("Nummer ausserhalb der Liste")
        return
    print(f"A  {left['recorded_at']}  {left['tool']}  ({left['provenance']})")
    print(f"B  {right['recorded_at']}  {right['tool']}  ({right['provenance']})\n")

    code_a = left["code"].get("combined")
    code_b = right["code"].get("combined")
    if code_a and code_b:
        same = "gleicher Codestand" if code_a == code_b else "UNTERSCHIEDLICHER Codestand"
        print(f"Code: {code_a} / {code_b} - {same}")
        if code_a != code_b:
            changed = [n for n, h in left["code"]["files"].items()
                       if right["code"]["files"].get(n) != h]
            print(f"      geaendert: {', '.join(changed) or '(nur Dateiliste)'}")
    else:
        print("Code: mindestens ein Lauf ist nachgetragen - nicht vergleichbar")

    keys = sorted({k for e in (left, right) for k, v in e["params"].items()
                   if left["params"].get(k) != right["params"].get(k)})
    if keys:
        print("\nAbweichende Parameter:")
        for k in keys:
            print(f"  {k}: {left['params'].get(k)!r}  ->  {right['params'].get(k)!r}")
    else:
        print("\nParameter identisch.")

    print("\nA:")
    for row in left["rows"]:
        print("  " + "  ".join(f"{k}={v}" for k, v in row.items()))
    print("B:")
    for row in right["rows"]:
        print("  " + "  ".join(f"{k}={v}" for k, v in row.items()))


if __name__ == "__main__":
    main()
