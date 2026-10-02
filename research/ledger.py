"""Follow and bill runs launched with `--spawn` (see research/modal_app.py).

    python research/ledger.py                 # status of the latest ledgers
    python research/ledger.py --record RUN_ID # append a finished run to BUDGET.md

Reads /vol/ledger/<run>.json from the Modal volume. A run's ledger is
rewritten after every batch, so a run that died shows its last heartbeat.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modal_app  # noqa: E402


def read(run_id: str) -> dict:
    return json.loads(b"".join(modal_app.vol.read_file(f"ledger/{run_id}.json")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", nargs="*", default=None, help="run ids to append to BUDGET.md")
    ap.add_argument("--latest", type=int, default=4)
    a = ap.parse_args()
    if a.record:
        billed = open(modal_app.BUDGET_MD, encoding="utf-8").read()
        for rid in a.record:
            if f"| {rid} |" in billed:
                print("already recorded:", rid)
                continue
            modal_app._record(read(rid))
        return
    names = sorted(e.path.split("/")[-1][:-5] for e in modal_app.vol.listdir("ledger"))
    for rid in names[-a.latest:]:
        r = read(rid)
        print(f"{rid}: {r['status']}, {r['seconds']:.0f} s, ${r['cost_usd']:.4f}")


if __name__ == "__main__":
    main()
