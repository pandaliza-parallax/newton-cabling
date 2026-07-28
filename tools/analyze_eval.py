#!/usr/bin/env python
"""Categorize closed-loop VLA eval rollouts into approach/align/insert/seat failure buckets.

Reads each evalrun_v2/ep_XXXX/trace.npz (motion trace from record_sbot_scene_gs.py --policy-server)
and classifies the episode by how far the policy got, using socket-frame insertion depth (y_sock)
and lateral offset (lat). Success = seated (y_sock reached the seat threshold).

Buckets (agreed taxonomy):
  seat/success  plug reached >= SEAT_MM into the socket (task done)
  approach      plug never neared the jack mouth (max depth stayed < APPROACH_MM below the mouth)
  align         reached the mouth but never entered (lateral > LAT_MM when nearest the mouth)
  insert        entered but jammed before seating (depth plateaued in (0, SEAT_MM))
  seat-slip     reached >= INSERT_MM but fell back / never held to seat

  python tools/analyze_eval.py --root evalrun_v2
"""
import argparse, glob, json, os
import numpy as np

SEAT_MM = 11.0      # full seat (matches --policy-server success gate, socket y >= 11mm)
INSERT_MM = 7.0     # "inserted" threshold (>=7mm in) from the eval metric
APPROACH_MM = -3.0  # plug is "at the mouth" when depth >= this (mm); below = still approaching
LAT_MM = 3.0        # lateral tolerance to enter the opening


def classify(tr):
    y = np.asarray(tr["y_sock"]) * 1000.0            # mm over time (mouth=0, + = deeper)
    lat = np.asarray(tr["lat"]) * 1000.0 if "lat" in tr else np.full_like(y, np.nan)
    ymax = float(y.max()); yfinal = float(y[-1])
    # lateral at the moment the plug was nearest the mouth (|depth| smallest)
    near_i = int(np.argmin(np.abs(y)))
    lat_near = float(lat[near_i]) if np.isfinite(lat[near_i]) else float("nan")

    if ymax >= SEAT_MM:
        bucket = "seat"                              # success
    elif ymax < APPROACH_MM:
        bucket = "approach"                          # never reached the mouth
    elif ymax < INSERT_MM:
        # reached the mouth region but didn't get past 7mm: align vs insert
        bucket = "align" if (np.isfinite(lat_near) and lat_near > LAT_MM) else "insert"
    else:
        bucket = "seat-slip"                         # got >=7mm but never fully seated/held
    return bucket, dict(depth_max_mm=round(ymax, 2), depth_final_mm=round(yfinal, 2),
                        lat_at_mouth_mm=round(lat_near, 2) if np.isfinite(lat_near) else None,
                        steps=len(y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="evalrun_v2")
    args = ap.parse_args()
    eps = sorted(glob.glob(os.path.join(args.root, "ep_*")))
    rows, counts = [], {b: 0 for b in ("seat", "seat-slip", "insert", "align", "approach")}
    for ep in eps:
        tp = os.path.join(ep, "trace.npz")
        if not os.path.exists(tp):
            continue
        tr = np.load(tp)
        if not len(tr["y_sock"]):
            continue
        bucket, info = classify(tr)
        counts[bucket] = counts.get(bucket, 0) + 1
        # cross-check against the renderer's own seated flag if present
        seated = None
        rj = os.path.join(ep, "result.json")
        if os.path.exists(rj):
            seated = json.load(open(rj)).get("seated")
        rows.append((os.path.basename(ep), bucket, info, seated))

    n = len(rows)
    print(f"\n===== closed-loop VLA eval: {n} episodes ({args.root}) =====")
    for name, bucket, info, seated in rows:
        flag = "" if seated is None or (seated == (bucket == "seat")) else "  [!seated-mismatch]"
        print(f"  {name}  {bucket:9s}  depth_max={info['depth_max_mm']:+6.1f}mm "
              f"final={info['depth_final_mm']:+6.1f}mm lat@mouth={info['lat_at_mouth_mm']}mm{flag}")
    print(f"\n  {'bucket':10s} {'count':>5s}  {'%':>5s}")
    for b in ("seat", "seat-slip", "insert", "align", "approach"):
        c = counts.get(b, 0)
        print(f"  {b:10s} {c:5d}  {100*c/n if n else 0:5.1f}")
    succ = counts.get("seat", 0)
    print(f"\n  SEAT SUCCESS: {succ}/{n} = {100*succ/n if n else 0:.1f}%")
    print(f"  (failures break down as align/insert/approach/seat-slip above)")


if __name__ == "__main__":
    main()
