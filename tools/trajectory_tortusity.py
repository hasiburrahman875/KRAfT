#!/usr/bin/env python3
# Compute agility and nonlinearity metrics over full datasets
# Metrics per track:
#  - tortuosity (path length over displacement)
#  - mean turn angle in degrees
#  - mean absolute angular velocity in degrees per second
#  - speed standard deviation in pixels per second
#  - heading entropy normalized in [0,1]
# Also computes a composite agility score via global robust normalization

import argparse, csv, math, re, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import configparser

# ---------------- I O ----------------

def load_seqinfo(seq_dir: Path):
    """Read frameRate from seqinfo.ini located in the sequence directory."""
    cfg = configparser.ConfigParser()
    ini = seq_dir / "seqinfo.ini"
    fps = 30.0
    if ini.exists():
        try:
            cfg.read(ini)
            s = cfg["Sequence"]
            fps = float(s.get("frameRate", fps))
        except Exception:
            pass
    return fps

def load_mot_gt_with_fps(gt_path: Path, min_len=5):
    """
    Load a MOTChallenge gt.txt and return:
     - dict: id -> np.ndarray[[f,x,y,w,h], ...] sorted by frame
     - fps for this sequence read from sibling seqinfo.ini
    Keeps tracks with >= min_len detections.
    """
    tracks = defaultdict(list)
    with open(gt_path, "r") as f:
        r = csv.reader(f)
        for row in r:
            if not row or row[0].startswith("#"):
                continue
            fr = int(float(row[0]))
            tid = int(float(row[1]))
            x = float(row[2]); y = float(row[3])
            w = float(row[4]); h = float(row[5])
            tracks[tid].append((fr, x, y, w, h))
    out = {}
    for tid, lst in tracks.items():
        if len(lst) < min_len:
            continue
        lst.sort(key=lambda t: t[0])
        out[tid] = np.asarray(lst, dtype=float)
    fps = load_seqinfo(gt_path.parent.parent)
    return out, fps

def centers_xy(arr):
    """Centers [N,2] from [[f,x,y,w,h], ...]"""
    x = arr[:,1] + 0.5*arr[:,3]
    y = arr[:,2] + 0.5*arr[:,4]
    return np.stack([x,y], axis=1)

# ---------------- Metrics ----------------

def tortuosity_xy(xy):
    """Total path over straight line displacement. Minimum 1.0."""
    n = xy.shape[0]
    if n < 3: return 1.0
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()
    disp = np.linalg.norm(xy[-1] - xy[0])
    return float(seg / max(disp, 1e-6))

def heading_angles(v):
    """Heading angles of velocity vectors in radians in [0, 2pi)."""
    ang = np.arctan2(v[:,1], v[:,0])  # [-pi, pi]
    ang = np.where(ang < 0, ang + 2*np.pi, ang)
    return ang

def mean_turn_angle_deg(xy):
    """Mean absolute turn angle in degrees between successive velocity vectors."""
    if xy.shape[0] < 3: return 0.0
    v = np.diff(xy, axis=0)
    norms = np.linalg.norm(v, axis=1)
    good = norms > 1e-6
    v = v[good]
    if v.shape[0] < 2: return 0.0
    a = []
    for i in range(v.shape[0]-1):
        v1, v2 = v[i], v[i+1]
        d = float(np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)))
        d = max(-1.0, min(1.0, d))
        a.append(math.degrees(math.acos(d)))
    return float(np.mean(a)) if a else 0.0

def mean_abs_angular_velocity_deg_per_s(xy, fps):
    """Mean absolute angular velocity in degrees per second."""
    if xy.shape[0] < 3: return 0.0
    v = np.diff(xy, axis=0)
    ang = heading_angles(v)  # radians
    dang = np.diff(ang)
    # unwrap to handle crossing pi boundary
    dang = (dang + np.pi) % (2*np.pi) - np.pi
    # per frame to per second
    av = np.mean(np.abs(dang)) * fps * 180.0 / np.pi
    return float(av)

def speed_std_px_per_s(xy, fps):
    """Standard deviation of speed in pixels per second."""
    if xy.shape[0] < 3: return 0.0
    v = np.diff(xy, axis=0)
    spd = np.linalg.norm(v, axis=1) * fps
    return float(np.std(spd))

def heading_entropy_norm(xy, bins=36):
    """Normalized entropy of heading distribution in [0,1]."""
    if xy.shape[0] < 4: return 0.0
    v = np.diff(xy, axis=0)
    norms = np.linalg.norm(v, axis=1)
    v = v[norms > 1e-6]
    if v.shape[0] < 5: return 0.0
    ang = heading_angles(v)  # [0,2pi)
    H, _ = np.histogram(ang, bins=bins, range=(0, 2*np.pi), density=False)
    p = H.astype(float) / max(1, H.sum())
    p = p[p > 0]
    if p.size == 0: return 0.0
    ent = -np.sum(p * np.log(p))
    ent_max = math.log(bins)
    return float(ent / ent_max) if ent_max > 0 else 0.0

def track_metrics(arr, fps):
    """Compute all metrics for one track array."""
    xy = centers_xy(arr)
    return {
        "tortuosity": tortuosity_xy(xy),
        "mean_turn_deg": mean_turn_angle_deg(xy),
        "mean_ang_vel_deg_s": mean_abs_angular_velocity_deg_per_s(xy, fps),
        "speed_std_px_s": speed_std_px_per_s(xy, fps),
        "heading_entropy": heading_entropy_norm(xy, bins=36),
        "len": xy.shape[0],
    }

# ---------------- Dataset scanning ----------------

def find_gt_files_folded(base_dir: Path, dataset_name: str):
    """
    Find gt files under *fold*/*test*/*/gt/gt.txt.
    Returns dict fold_label -> list[Path].
    """
    fold_map = defaultdict(list)
    # robust recursive search
    for gt in base_dir.rglob("**/*fold*/*test*/*/gt/gt.txt"):
        parts = gt.parts
        fold_dir = next((p for p in parts if "fold" in p.lower()), None)
        if fold_dir is None:
            continue
        m = re.search(r"fold[-_]?(\d+)", fold_dir, re.IGNORECASE)
        fold_label = f"fold{m.group(1)}" if m else fold_dir
        fold_map[fold_label].append(gt)
    if not fold_map:
        raise FileNotFoundError(f"[{dataset_name}] no folded test gt files under {base_dir}")
    for k in list(fold_map.keys()):
        fold_map[k] = sorted(fold_map[k])
    return dict(sorted(fold_map.items()))

def find_gt_files_flat(test_root: Path, dataset_name: str):
    """Find gt files under <test_root>/<seq>/gt/gt.txt."""
    paths = sorted(test_root.glob("*/gt/gt.txt"))
    if not paths:
        paths = sorted(test_root.rglob("**/*/gt/gt.txt"))
    if not paths:
        raise FileNotFoundError(f"[{dataset_name}] no test gt files under {test_root}")
    return {"test": paths}

# ---------------- Summaries and normalization ----------------

def robust_normalize(values, clip=True):
    """
    Robust normalization using global median and IQR.
    Returns normalized array in approx [0,1] after clipping.
    """
    v = np.asarray(values, dtype=float)
    med = np.median(v)
    q1 = np.percentile(v, 25)
    q3 = np.percentile(v, 75)
    iqr = max(1e-9, q3 - q1)
    z = (v - med) / iqr  # robust z
    # map to 0..1 via sigmoid-like transform
    norm = 1.0 / (1.0 + np.exp(-z))
    if clip:
        norm = np.clip(norm, 0.0, 1.0)
    return norm, {"median": float(med), "q1": float(q1), "q3": float(q3)}

def summarize_set(name, tracks_metrics):
    """
    tracks_metrics: list of dict with metrics for all tracks in a set
    return dict with summary stats
    """
    if not tracks_metrics:
        return {"name": name, "ids": 0}
    keys = ["tortuosity","mean_turn_deg","mean_ang_vel_deg_s","speed_std_px_s","heading_entropy","agility_score"]
    res = {"name": name, "ids": len(tracks_metrics)}
    for k in keys:
        arr = np.array([m[k] for m in tracks_metrics], dtype=float)
        res.update({
            f"{k}_mean": float(np.mean(arr)),
            f"{k}_median": float(np.median(arr)),
            f"{k}_p25": float(np.percentile(arr,25)),
            f"{k}_p75": float(np.percentile(arr,75)),
            f"{k}_max": float(np.max(arr)),
        })
    return res

def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser(description="Agility metrics over full datasets with composite score.")
    ap.add_argument("--w2c_root", type=str, default=
        "/cluster/pixstor/madrias-lab/Hasibur/AT/YOLOv12-BoT-SORT-ReID-w2c/BoT-SORT/TrackEval/data/gt/mot_challenge/UAVSwarm",
        help="Root containing w2c folds")
    ap.add_argument("--uav_test_root", type=str, default=
        "/cluster/pixstor/madrias-lab/Hasibur/AT/YOLOv12-BoT-SORT-ReID/BoT-SORT/TrackEval/data/gt/mot_challenge/UAVSwarm/UAVSwarm-test",
        help="UAVSwarm test root no folds")
    ap.add_argument("--muav_root", type=str, default=
        "/cluster/pixstor/madrias-lab/Hasibur/AT/YOLOv12-BoT-SORT-ReID-uav/BoT-SORT/TrackEval/data/gt/mot_challenge",
        help="Root containing MUAV folds")
    ap.add_argument("--min_len", type=int, default=15, help="minimum detections per ID")
    ap.add_argument("--out_dir", type=str, default="agility_stats")
    args = ap.parse_args()

    # collect per track metrics across all datasets for global normalization
    all_tracks = []  # list of dicts with metrics + dataset labels
    per_set_tracks = defaultdict(list)

    # W2C folded
    w2c_sets = find_gt_files_folded(Path(args.w2c_root), "UAVSwarm W2C")
    print("\n=== UAVSwarm W2C per fold ===")
    for fold, gts in w2c_sets.items():
        for gt in gts:
            tracks, fps = load_mot_gt_with_fps(gt, min_len=args.min_len)
            for tid, arr in tracks.items():
                m = track_metrics(arr, fps)
                rec = {"dataset":"UAVSwarm-W2C","set":fold,"seq":gt.parent.parent.name,"id":tid, **m}
                all_tracks.append(rec)
                per_set_tracks[f"UAVSwarm-W2C:{fold}"].append({**m})
    # UAVSwarm flat
    uav_sets = find_gt_files_flat(Path(args.uav_test_root), "UAVSwarm")
    print("\n=== UAVSwarm test ===")
    for split, gts in uav_sets.items():
        for gt in gts:
            tracks, fps = load_mot_gt_with_fps(gt, min_len=args.min_len)
            for tid, arr in tracks.items():
                m = track_metrics(arr, fps)
                rec = {"dataset":"UAVSwarm","set":split,"seq":gt.parent.parent.name,"id":tid, **m}
                all_tracks.append(rec)
                per_set_tracks[f"UAVSwarm:{split}"].append({**m})
    # MUAV folded
    muav_sets = find_gt_files_folded(Path(args.muav_root), "MUAV")
    print("\n=== MUAV per fold ===")
    for fold, gts in muav_sets.items():
        for gt in gts:
            tracks, fps = load_mot_gt_with_fps(gt, min_len=args.min_len)
            for tid, arr in tracks.items():
                m = track_metrics(arr, fps)
                rec = {"dataset":"MUAV","set":fold,"seq":gt.parent.parent.name,"id":tid, **m}
                all_tracks.append(rec)
                per_set_tracks[f"MUAV:{fold}"].append({**m})

    if not all_tracks:
        print("no tracks found, check paths and min_len", file=sys.stderr)
        sys.exit(1)

    # global robust normalization for composite agility score
    T = np.array([r["tortuosity"] for r in all_tracks], dtype=float)
    A = np.array([r["mean_turn_deg"] for r in all_tracks], dtype=float)
    V = np.array([r["mean_ang_vel_deg_s"] for r in all_tracks], dtype=float)
    S = np.array([r["speed_std_px_s"] for r in all_tracks], dtype=float)
    E = np.array([r["heading_entropy"] for r in all_tracks], dtype=float)

    Tn, Tstats = robust_normalize(T)
    An, Astats = robust_normalize(A)
    Vn, Vstats = robust_normalize(V)
    Sn, Sstats = robust_normalize(S)
    En, Estats = robust_normalize(E)

    # weights can be tuned; equal weights by default
    weights = np.array([1,1,1,1,1], dtype=float)
    norm_stack = np.stack([Tn, An, Vn, Sn, En], axis=1)
    composite = (norm_stack * weights).sum(axis=1) / weights.sum()

    # attach to all_tracks and per set
    for i, r in enumerate(all_tracks):
        r["agility_score"] = float(composite[i])

    # print per set summaries
    summaries = []
    print("\n=== per set summaries ===")
    for set_name, metrics_list in per_set_tracks.items():
        # attach agility score for each record by recomputing from all_tracks mapping
        # simpler approach: pull from all_tracks
        ids_in_set = [i for i, r in enumerate(all_tracks) if f"{r['dataset']}:{r['set']}" == set_name]
        for idx in ids_in_set:
            # ensure metrics_list has agility_score
            pass
        # build list with agility_score
        ml = []
        for i, r in enumerate(all_tracks):
            if f"{r['dataset']}:{r['set']}" == set_name:
                ml.append({k:r[k] for k in ["tortuosity","mean_turn_deg","mean_ang_vel_deg_s","speed_std_px_s","heading_entropy","agility_score"]})
        summ = summarize_set(set_name, ml)
        summaries.append(summ)
        print(f"{set_name}: ids {summ['ids']}, "
              f"tort median {summ['tortuosity_median']:.3f}, "
              f"turn median {summ['mean_turn_deg_median']:.3f}, "
              f"ang vel median {summ['mean_ang_vel_deg_s_median']:.3f}, "
              f"spd std median {summ['speed_std_px_s_median']:.3f}, "
              f"entropy median {summ['heading_entropy_median']:.3f}, "
              f"agility median {summ['agility_score_median']:.3f}")

    # overall per dataset summaries
    dataset_groups = defaultdict(list)
    for r in all_tracks:
        dataset_groups[r["dataset"]].append({k:r[k] for k in ["tortuosity","mean_turn_deg","mean_ang_vel_deg_s","speed_std_px_s","heading_entropy","agility_score"]})

    print("\n=== overall dataset summaries ===")
    ds_summ_rows = []
    for ds, ml in dataset_groups.items():
        summ = summarize_set(ds, ml)
        ds_summ_rows.append(summ)
        print(f"{ds}: ids {summ['ids']}, "
              f"tort median {summ['tortuosity_median']:.3f}, "
              f"turn median {summ['mean_turn_deg_median']:.3f}, "
              f"ang vel median {summ['mean_ang_vel_deg_s_median']:.3f}, "
              f"spd std median {summ['speed_std_px_s_median']:.3f}, "
              f"entropy median {summ['heading_entropy_median']:.3f}, "
              f"agility median {summ['agility_score_median']:.3f}")

    # write CSVs
    out_dir = Path(args.out_dir)
    # all tracks detailed
    fieldnames = ["dataset","set","seq","id","len",
                  "tortuosity","mean_turn_deg","mean_ang_vel_deg_s","speed_std_px_s","heading_entropy","agility_score"]
    write_csv(out_dir / "all_tracks_metrics.csv", all_tracks, fieldnames)

    # per set summaries
    if summaries:
        write_csv(out_dir / "per_set_summary.csv", summaries, list(summaries[0].keys()))
    # per dataset summaries
    if ds_summ_rows:
        write_csv(out_dir / "per_dataset_summary.csv", ds_summ_rows, list(ds_summ_rows[0].keys()))

    # print normalization references
    print("\n=== global robust normalization refs ===")
    print(f"Tortuosity median {Tstats['median']:.3f} q1 {Tstats['q1']:.3f} q3 {Tstats['q3']:.3f}")
    print(f"Turn deg median {Astats['median']:.3f} q1 {Astats['q1']:.3f} q3 {Astats['q3']:.3f}")
    print(f"Ang vel deg s median {Vstats['median']:.3f} q1 {Vstats['q1']:.3f} q3 {Vstats['q3']:.3f}")
    print(f"Speed std px s median {Sstats['median']:.3f} q1 {Sstats['q1']:.3f} q3 {Sstats['q3']:.3f}")
    print(f"Entropy median {Estats['median']:.3f} q1 {Estats['q1']:.3f} q3 {Estats['q3']:.3f}")
    print("CSV written to", out_dir.resolve())

if __name__ == "__main__":
    main()
