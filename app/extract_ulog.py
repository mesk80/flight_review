"""
extract_ulog.py

Convert every PX4 ULog file in a folder into one CSV file.

Output columns (one row per vehicle_local_position sample):
    t            time in seconds, same origin for all columns
    lat, lon     degrees
    alt          meters (AMSL)
    N, E, D      meters (PX4 local NED frame)
    ground_speed m/s, computed from vx, vy of the local position
    roll, pitch, yaw   radians (from the attitude quaternion)

All signals are interpolated onto the vehicle_local_position timestamps,
and only the time range covered by all three topics is kept.

Usage:
    python extract_ulog.py <ulog_dir> <output_dir>
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pyulog import ULog

LOCAL = "vehicle_local_position"
GLOBAL = "vehicle_global_position"
ATTITUDE = "vehicle_attitude"


def get_topic(ulog: ULog, name: str) -> dict:
    """Return the data of the first instance of a topic."""
    for d in ulog.data_list:
        if d.name == name and d.multi_id == 0:
            return d.data
    raise KeyError(f"topic '{name}' not found")


def sorted_unique(ts: np.ndarray, *arrays: np.ndarray):
    """Sort by time and drop repeated timestamps."""
    ts, idx = np.unique(ts, return_index=True)
    return (ts, *[a[idx] for a in arrays])


def quat_to_euler(q: np.ndarray):
    """Quaternion [w, x, y, z] to roll, pitch, yaw (radians, ZYX order)."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def extract(path: Path) -> pd.DataFrame:
    """Read one ULog and return the aligned data as a DataFrame."""
    ulog = ULog(str(path))
    loc, glob, att = get_topic(ulog, LOCAL), get_topic(ulog, GLOBAL), get_topic(ulog, ATTITUDE)

    # Local position (NED) and ground speed
    t_loc, n, e, d, vx, vy = sorted_unique(
        loc["timestamp"], loc["x"], loc["y"], loc["z"], loc["vx"], loc["vy"]
    )

    # Global position
    t_glob, lat, lon, alt = sorted_unique(
        glob["timestamp"], glob["lat"], glob["lon"], glob["alt"]
    )

    # Attitude: quaternion -> Euler angles
    q = np.stack([att[f"q[{i}]"] for i in range(4)], axis=1)
    t_att, q = sorted_unique(att["timestamp"], q)
    roll, pitch, yaw = quat_to_euler(q)
    yaw = np.unwrap(yaw)  # avoid jumps at +-pi while interpolating

    # Common time range covered by all topics
    t_start = max(t_loc[0], t_glob[0], t_att[0])
    t_end = min(t_loc[-1], t_glob[-1], t_att[-1])
    mask = (t_loc >= t_start) & (t_loc <= t_end)
    if mask.sum() < 2:
        raise ValueError("topics do not overlap in time")
    base = t_loc[mask]

    def on_base(t_src, values):
        return np.interp(base, t_src, values)

    return pd.DataFrame({
        "t": (base - base[0]) * 1e-6,  # microseconds -> seconds, starts at 0
        "lat": on_base(t_glob, lat),
        "lon": on_base(t_glob, lon),
        "alt": on_base(t_glob, alt),
        "N": n[mask],
        "E": e[mask],
        "D": d[mask],
        "ground_speed": np.hypot(vx[mask], vy[mask]),
        "roll": on_base(t_att, roll),
        "pitch": on_base(t_att, pitch),
        "yaw": (on_base(t_att, yaw) + np.pi) % (2 * np.pi) - np.pi,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract flight data from ULog files.")
    parser.add_argument("ulog_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in args.ulog_dir.iterdir()
                   if p.suffix.lower() in (".ulg", ".ulog"))

    # Original names are irregular, so output files get ordered IDs.
    # The mapping is saved in metadata.csv.
    meta = []
    for i, path in enumerate(files, start=1):
        flight_id = f"flight_{i:02d}"
        try:
            df = extract(path)
        except Exception as err:  # keep going if one log is bad
            print(f"[skip] {path.name}: {err}")
            continue
        df.to_csv(args.out_dir / f"{flight_id}.csv", index=False)
        meta.append({
            "flight_id": flight_id,
            "source_file": path.name,
            "n_rows": len(df),
            "duration_s": df["t"].iloc[-1],
            "median_dt_s": float(np.median(np.diff(df["t"]))),
        })
        print(f"[ok]   {path.name} -> {flight_id}.csv ({len(df)} rows)")

    pd.DataFrame(meta).to_csv(args.out_dir / "metadata.csv", index=False)


if __name__ == "__main__":
    main()
