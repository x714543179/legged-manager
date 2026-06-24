#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import sys
import time
from pathlib import Path

import numpy as np

try:
    import viser
except ImportError as exc:  # pragma: no cover
    raise ImportError("viser is required. Install with: python -m pip install viser") from exc


G1_FULL_BODY_NAMES = [
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]

G1_AMP_EDGES = [
    ("pelvis", "left_hip_roll_link"),
    ("left_hip_roll_link", "left_knee_link"),
    ("left_knee_link", "left_ankle_roll_link"),
    ("pelvis", "right_hip_roll_link"),
    ("right_hip_roll_link", "right_knee_link"),
    ("right_knee_link", "right_ankle_roll_link"),
    ("pelvis", "left_shoulder_roll_link"),
    ("left_shoulder_roll_link", "left_elbow_link"),
    ("left_elbow_link", "left_wrist_yaw_link"),
    ("pelvis", "right_shoulder_roll_link"),
    ("right_shoulder_roll_link", "right_elbow_link"),
    ("right_elbow_link", "right_wrist_yaw_link"),
]


def _repo_root() -> Path:
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / "rsl_rl").is_dir() and (parent / "whole_body_tracking").is_dir():
            return parent
    raise RuntimeError(f"Could not find workspace root from {path}")


def _add_repo_paths() -> None:
    repo_root = _repo_root()
    for path in (repo_root / "rsl_rl",):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def _load_g1_amp_defaults() -> tuple[list[str], str]:
    cfg_path = (
        _repo_root()
        / "whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py"
    )
    tree = ast.parse(cfg_path.read_text(encoding="utf-8"), filename=str(cfg_path))
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"G1_AMP_BODY_NAMES", "G1_AMP_ANCHOR_BODY_NAME"}:
                    values[target.id] = ast.literal_eval(node.value)
    return list(values["G1_AMP_BODY_NAMES"]), str(values["G1_AMP_ANCHOR_BODY_NAME"])


def _read_names_file(path: str | None) -> list[str] | None:
    if path is None:
        return None
    names = []
    for line in Path(path).expanduser().read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            names.append(line)
    return names


def _edge_indices(body_names: list[str]) -> np.ndarray:
    name_to_index = {name: index for index, name in enumerate(body_names)}
    edges = [(name_to_index[a], name_to_index[b]) for a, b in G1_AMP_EDGES if a in name_to_index and b in name_to_index]
    return np.asarray(edges, dtype=np.int64)


def _colors(count: int, color: tuple[int, int, int]) -> np.ndarray:
    return np.tile(np.asarray(color, dtype=np.uint8), (count, 1))


def _segment_colors(count: int, color: tuple[int, int, int]) -> np.ndarray:
    return np.tile(np.asarray(color, dtype=np.uint8), (count, 2, 1))


def parse_args() -> argparse.Namespace:
    default_body_names, default_anchor_name = _load_g1_amp_defaults()
    parser = argparse.ArgumentParser(description="Visualize preprocessed expert data produced by AMPLoader.")
    parser.add_argument("motion_file", help="Single npz or directory passed to AMPLoader.")
    parser.add_argument("--body_names_file", default=None, help="Optional txt file overriding AMP body order.")
    parser.add_argument("--anchor_name", default=default_anchor_name)
    parser.add_argument("--body_names", nargs="*", default=None, help="Override AMP body order directly.")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--motion_index", type=int, default=0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.set_defaults(default_body_names=default_body_names)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _add_repo_paths()
    from rsl_rl.algorithms.plugins.amp.motion_loader import AMPLoader

    body_names = args.body_names or _read_names_file(args.body_names_file) or list(args.default_body_names)
    anchor_name = args.anchor_name
    if anchor_name not in body_names:
        raise ValueError(f"anchor_name '{anchor_name}' must be included in body_names for the current AMPLoader.")

    loader = AMPLoader(
        motion_file=args.motion_file,
        body_names=body_names,
        anchor_name=anchor_name,
        all_body_names=G1_FULL_BODY_NAMES,
        device="cpu",
    )

    motion_index = int(np.clip(args.motion_index, 0, len(loader._body_pos_b_list) - 1))
    pos = loader._body_pos_b_list[motion_index].cpu().numpy()
    fps = float(np.asarray(loader.fps).reshape(-1)[0])
    edges = _edge_indices(body_names)
    if edges.size == 0:
        raise ValueError("No skeleton edges could be built from the selected AMP body_names.")

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.add_grid("/ground", width=4.0, height=4.0)

    with server.gui.add_folder("amp_motion_loader"):
        playing = server.gui.add_checkbox("playing", initial_value=True)
        speed = server.gui.add_slider("speed", min=0.05, max=4.0, step=0.05, initial_value=float(args.speed))
        frame_slider = server.gui.add_slider("frame", min=0, max=pos.shape[0] - 1, step=1, initial_value=0)
        server.gui.add_text("motion", initial_value=loader.motion_names[motion_index])
        server.gui.add_text("anchor", initial_value=anchor_name)
        server.gui.add_text("body_order", initial_value=", ".join(body_names))

    frame = server.scene.add_frame("/amp/root", position=(0.0, 0.0, 0.0), show_axes=True)
    points = server.scene.add_point_cloud(
        "/amp/points",
        points=pos[0],
        colors=_colors(pos.shape[1], (255, 190, 60)),
        point_size=0.035,
    )
    lines = server.scene.add_line_segments(
        "/amp/skeleton",
        points=pos[0, edges],
        colors=_segment_colors(edges.shape[0], (80, 180, 255)),
        line_width=3.0,
    )
    labels = []
    for index, name in enumerate(body_names):
        label = server.scene.add_label(f"/amp/labels/{index:02d}", text=f"{index}: {name}")
        label.position = tuple(pos[0, index] + np.asarray([0.0, 0.0, 0.04], dtype=np.float32))
        labels.append(label)

    print(f"[INFO] Visualizing AMPLoader output for: {loader.motion_names[motion_index]}")
    print(f"[INFO] AMP body order: {body_names}")
    print(f"[INFO] Open http://localhost:{args.port}")

    cursor = 0.0
    last_time = time.time()
    try:
        while True:
            now = time.time()
            dt = now - last_time
            last_time = now
            if playing.value:
                cursor = (cursor + dt * fps * float(speed.value)) % pos.shape[0]
                frame_slider.value = int(cursor)
            else:
                cursor = float(frame_slider.value)
            frame_id = int(cursor) % pos.shape[0]
            body = pos[frame_id]
            with server.atomic():
                frame.position = tuple(body[body_names.index(anchor_name)].tolist())
                points.points = body
                lines.points = body[edges]
                for index, label in enumerate(labels):
                    label.position = tuple((body[index] + np.asarray([0.0, 0.0, 0.04])).tolist())
            time.sleep(1.0 / 30.0)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
