#!/usr/bin/env python3
"""
Extract training features from an in-cabin video dataset.

Designed for datasets like NTHU/UTA where class labels can be inferred
from parent folder names or filename patterns.
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import deque

import cv2
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from vision.vision_engine import create_vision_processor


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
FEATURE_COLUMNS = [
    "left_ear",
    "right_ear",
    "avg_ear",
    "ear_variance",
    "blink_frequency",
    "eye_closure_duration",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "mar",
    "yawn_frequency",
    "label",
    "source_path",
    "frames_used",
]


def load_label_map(path):
    if not path:
        return {}
    with open(path, "r") as f:
        return json.load(f)


def infer_label(path, label_map):
    lower_path = path.lower()
    for pattern, label in label_map.items():
        if re.search(pattern, lower_path):
            return label
    # fallback: parent directory as label
    return os.path.basename(os.path.dirname(path)).lower()


def iter_video_files(root):
    for base, _, files in os.walk(root):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext in VIDEO_EXTS:
                yield os.path.join(base, name)


def clip_to_features(video_path, label, vision, frame_stride, max_frames):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    ear_values = []
    mar_values = []
    head_pitch = []
    head_yaw = []
    head_roll = []
    blink_count = 0
    yawn_count = 0
    frames_used = 0

    prev_eye_closed = False
    eye_closed_streak = 0
    yawn_active = False
    yawn_streak = 0
    idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx % frame_stride != 0:
                continue

            result = vision.process_frame(frame)
            if not result["face_detected"]:
                continue

            ea = result["eye_analysis"]
            ma = result["mouth_analysis"]
            hp = result["head_pose"]

            avg_ear = float(ea.get("avg_ear", 0.0))
            left_ear = float(ea.get("left_ear", 0.0))
            right_ear = float(ea.get("right_ear", 0.0))
            mar = float(ma.get("mar", 0.0))

            ear_values.append((left_ear, right_ear, avg_ear))
            mar_values.append(mar)

            if hp:
                head_pitch.append(float(hp.get("pitch", 0.0)))
                head_yaw.append(float(hp.get("yaw", 0.0)))
                head_roll.append(float(hp.get("roll", 0.0)))
            else:
                head_pitch.append(0.0)
                head_yaw.append(0.0)
                head_roll.append(0.0)

            # Blink counting from EAR transitions
            eye_closed = avg_ear < 0.22
            if eye_closed:
                eye_closed_streak += 1
            if prev_eye_closed and not eye_closed and 1 <= eye_closed_streak <= 12:
                blink_count += 1
                eye_closed_streak = 0
            prev_eye_closed = eye_closed

            # Yawn counting from MAR streak
            yawn_now = mar > 0.5
            if yawn_now:
                yawn_streak += 1
                if not yawn_active and yawn_streak >= 3:
                    yawn_count += 1
                    yawn_active = True
            else:
                yawn_streak = 0
                yawn_active = False

            frames_used += 1
            if frames_used >= max_frames:
                break
    finally:
        cap.release()

    if frames_used == 0:
        return None

    left_arr = np.array([x[0] for x in ear_values], dtype=float)
    right_arr = np.array([x[1] for x in ear_values], dtype=float)
    avg_arr = np.array([x[2] for x in ear_values], dtype=float)
    mar_arr = np.array(mar_values, dtype=float)

    # Approximate rates per minute for consistency with runtime features
    # Assuming effective processed fps around 30/frame_stride.
    effective_fps = max(1.0, 30.0 / frame_stride)
    duration_sec = frames_used / effective_fps
    duration_min = max(duration_sec / 60.0, 1e-6)

    eye_closure_duration = float(np.mean((avg_arr < 0.22).astype(float)) * duration_sec)
    row = {
        "left_ear": float(np.mean(left_arr)),
        "right_ear": float(np.mean(right_arr)),
        "avg_ear": float(np.mean(avg_arr)),
        "ear_variance": float(np.var(avg_arr)),
        "blink_frequency": float(blink_count / duration_min),
        "eye_closure_duration": eye_closure_duration,
        "head_pitch": float(np.mean(head_pitch)),
        "head_yaw": float(np.mean(head_yaw)),
        "head_roll": float(np.mean(head_roll)),
        "mar": float(np.mean(mar_arr)),
        "yawn_frequency": float(yawn_count / duration_min),
        "label": label,
        "source_path": video_path,
        "frames_used": frames_used,
    }
    return row


def main():
    parser = argparse.ArgumentParser(description="Extract feature CSV from in-cabin video dataset.")
    parser.add_argument("--dataset-root", required=True, help="Root folder containing videos.")
    parser.add_argument("--output", default="data/v1/features_dataset.csv", help="Output feature CSV path.")
    parser.add_argument(
        "--label-map",
        default="",
        help="Optional JSON of regex->label mappings for robust label inference.",
    )
    parser.add_argument("--frame-stride", type=int, default=3, help="Use every Nth frame.")
    parser.add_argument("--max-frames-per-clip", type=int, default=450, help="Max processed frames per clip.")
    parser.add_argument("--limit-clips", type=int, default=0, help="Optional clip limit for quick runs.")
    args = parser.parse_args()

    if not os.path.exists(args.dataset_root):
        raise ValueError(f"Dataset root does not exist: {args.dataset_root}")

    label_map = load_label_map(args.label_map)
    vision = create_vision_processor()
    rows = []

    videos = list(iter_video_files(args.dataset_root))
    if args.limit_clips > 0:
        videos = videos[: args.limit_clips]

    for i, video_path in enumerate(videos, start=1):
        label = infer_label(video_path, label_map)
        row = clip_to_features(
            video_path=video_path,
            label=label,
            vision=vision,
            frame_stride=max(1, args.frame_stride),
            max_frames=max(1, args.max_frames_per_clip),
        )
        if row:
            rows.append(row)
        if i % 20 == 0:
            print(f"Processed {i}/{len(videos)} videos, usable rows: {len(rows)}")

    if not rows:
        raise ValueError("No usable feature rows extracted. Check dataset format and landmark model path.")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FEATURE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
