#!/usr/bin/env python3
import argparse, re
from pathlib import Path
import pandas as pd
import numpy as np
from itertools import combinations

PLAYER_COORD_RE = re.compile(r"^player_(\d+)_(x|y)$")

def parse_player_cols(df):
    """
    Returns a dict: player_id -> {'x': colname, 'y': colname}
    based on columns like player_3_x, player_3_y.
    """
    mapping = {}
    for col in df.columns:
        m = PLAYER_COORD_RE.match(col)
        if not m: 
            continue
        pid, axis = m.groups()
        d = mapping.setdefault(int(pid), {})
        d[axis] = col
    # keep only players that have both x and y columns
    return {pid: v for pid, v in mapping.items() if 'x' in v and 'y' in v}

def per_frame_positions(row, player_map):
    """Return {pid: (x,y)} for players present in this frame row (both x,y not empty)."""
    positions = {}
    for pid, cols in player_map.items():
        x = row.get(cols['x'], '')
        y = row.get(cols['y'], '')
        if x == '' or y == '' or pd.isna(x) or pd.isna(y):
            continue
        try:
            positions[pid] = (float(x), float(y))
        except Exception:
            # non-numeric, skip
            continue
    return positions

def summarize_file(csv_path, out_long=None):
    df = pd.read_csv(csv_path)
    # Required base columns
    base_cols = set(df.columns.str.lower())
    frame_col = 'frame_number' if 'frame_number' in df.columns else None
    catcher_col = 'catcher_player_id' if 'catcher_player_id' in df.columns else None
    catchdist_col = 'catch_distance_px' if 'catch_distance_px' in df.columns else None

    player_map = parse_player_cols(df)
    if not player_map:
        raise ValueError("No player_{id}_x / player_{id}_y columns found.")

    # Catch detection: any frame with catcher_player_id >= 0
    has_catch = None
    first_catch_frame = None
    catcher_ids = set()
    if catcher_col and catcher_col in df.columns:
        mask = pd.to_numeric(df[catcher_col], errors='coerce').fillna(-1) >= 0
        if mask.any():
            has_catch = True
            first_catch_frame = int(df.loc[mask, frame_col].iloc[0]) if frame_col else None
            catcher_ids = set(int(x) for x in pd.to_numeric(df.loc[mask, catcher_col], errors='coerce').dropna().unique())
        else:
            has_catch = False

    # Pairwise distances per frame
    frames_with_two = 0
    n_frames = 0
    all_dists = []
    long_rows = []

    for _, row in df.iterrows():
        n_frames += 1
        positions = per_frame_positions(row, player_map)
        if len(positions) >= 2:
            frames_with_two += 1
            # compute pairwise distances
            items = list(positions.items())  # [(pid, (x,y)), ...]
            for (pid1, p1), (pid2, p2) in combinations(items, 2):
                d = np.linalg.norm(np.array(p1) - np.array(p2))
                all_dists.append(d)
                if out_long:
                    long_rows.append({
                        'frame_number': int(row[frame_col]) if frame_col else None,
                        'player_a': pid1, 'player_b': pid2,
                        'ax': p1[0], 'ay': p1[1], 'bx': p2[0], 'by': p2[1],
                        'distance_px': float(d)
                    })

    if out_long and long_rows:
        out_long.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(long_rows).to_csv(out_long, index=False)

    if all_dists:
        dist_min = float(np.min(all_dists))
        dist_mean = float(np.mean(all_dists))
        dist_max = float(np.max(all_dists))
    else:
        dist_min = dist_mean = dist_max = None

    return {
        'file': str(csv_path),
        'num_frames': n_frames,
        'frames_with_>=2_players': frames_with_two,
        'prop_frames_with_>=2': frames_with_two / max(n_frames, 1),
        'has_catch': has_catch,
        'first_catch_frame': first_catch_frame,
        'catcher_ids': ';'.join(map(str, sorted(catcher_ids))) if catcher_ids else '',
        'min_distance_px': dist_min,
        'mean_distance_px': dist_mean,
        'max_distance_px': dist_max,
        'players_columns_found': ';'.join([f'player_{pid}_x,player_{pid}_y' for pid in sorted(player_map.keys())])
    }

def main():
    ap = argparse.ArgumentParser(description="Validate features from SimpleOutfielderTracker CSVs")
    ap.add_argument('--inputs', nargs='+', required=True, help='CSV file(s) produced by export_simple_csv')
    ap.add_argument('--out', default='deliverables/joshua/week2/validation_report.csv', help='Path for summary CSV')
    ap.add_argument('--emit-long', action='store_true', help='Also emit per-frame pairwise distances CSV (long-form)')
    args = ap.parse_args()

    outp = Path(args.out)
    rows = []
    for p in args.inputs:
        p = Path(p)
        if not p.exists():
            print(f"[WARN] missing: {p}")
            continue
        long_path = None
        if args.emit_long:
            long_path = outp.with_name(outp.stem + f'__{p.stem}_pairwise.csv')
        summary = summarize_file(p, long_path)
        rows.append(summary)

    if not rows:
        print("No inputs processed.")
        return

    outp.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(outp, index=False)
    print(f"Wrote summary → {outp}")
    for r in rows:
        print(f"- {Path(r['file']).name}: catch={r['has_catch']} "
              f"two_frames={r['frames_with_>=2_players']}/{r['num_frames']} "
              f"min_dist={r['min_distance_px']}")
if __name__ == '__main__':
    main()
