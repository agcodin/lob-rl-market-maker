#!/usr/bin/env python
"""Inject the recorded episode data into the dashboard HTML template."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def skew_table(trace: dict) -> list[dict]:
    """Mean quote offsets bucketed by the inventory held going into the step."""
    inv = np.array(trace["inventory"][:-1])
    bid = np.array(trace["delta_bid"][1:])
    ask = np.array(trace["delta_ask"][1:])
    buckets = [(-1e9, -10, "short 10+"), (-10, -3, "short 3-10"), (-3, 3, "flat"),
               (3, 10, "long 3-10"), (10, 1e9, "long 10+")]
    out = []
    for lo, hi, label in buckets:
        m = (inv >= lo) & (inv < hi)
        if m.sum() == 0:
            continue
        out.append({"label": label, "n": int(m.sum()),
                    "bid": round(float(bid[m].mean()), 2),
                    "ask": round(float(ask[m].mean()), 2)})
    corr = float(np.corrcoef(inv, bid - ask)[0, 1])
    return out, round(corr, 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="runs/dashboard_data.json")
    ap.add_argument("--template", default="scripts/dashboard_template.html")
    ap.add_argument("--multi", default="runs/multi_data.json",
                    help="competitive results; the sections are dropped if absent")
    ap.add_argument("--arena", default="runs/arena_data.json",
                    help="overnight + intervention results; section dropped if absent")
    ap.add_argument("--informed", default="runs/informed_data.json")
    ap.add_argument("--out", default="runs/dashboard.html")
    args = ap.parse_args()

    d = json.loads(Path(args.data).read_text())
    ppo = d["episodes"]["ppo"]
    skew, corr = skew_table(ppo["trace"])

    keep = ("mid", "best_bid", "best_ask", "quote_bid", "quote_ask", "inventory",
            "equity", "delta_bid", "delta_ask", "spread")
    payload = {
        "seed": d["seed"],
        "steps": d["steps"],
        "config": d["config"],
        "ppo": {
            "trace": {k: ppo["trace"][k] for k in keep},
            "book": ppo["book"],
            "fills": ppo["fills"],
            "metrics": {k: round(v, 3) for k, v in ppo["metrics"].items()},
        },
        "others": {
            name: {
                "inventory": ep["trace"]["inventory"],
                "equity": ep["trace"]["equity"],
                "metrics": {k: round(v, 3) for k, v in ep["metrics"].items()},
            }
            for name, ep in d["episodes"].items() if name != "ppo"
        },
        "aggregate": d["aggregate"],
        "skew": {"buckets": skew, "corr": corr},
        "training": [
            {"steps": r["steps"], "ret": round(r["mean_return"], 1),
             "inv": round(r["mean_abs_inv"], 2), "trades": round(r["mean_trades"], 1),
             "equity": round(r["mean_equity"], 1)}
            for r in d["training"][::2]
        ],
    }

    mp = Path(args.multi)
    payload["multi"] = json.loads(mp.read_text()) if mp.exists() else None
    if payload["multi"] is None:
        print(f"[warn] {mp} missing -- building without the competition sections")

    ar = Path(args.arena)
    payload["arena"] = json.loads(ar.read_text()) if ar.exists() else None

    ip = Path(args.informed)
    payload["informed"] = json.loads(ip.read_text()) if ip.exists() else None

    html = Path(args.template).read_text()
    out = html.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
    Path(args.out).write_text(out)
    print(f"wrote {args.out} ({Path(args.out).stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
