#!/usr/bin/env python3
"""
Dataset collection utility for Driver Monitor System.

Collects labeled short clips for:
  - alert
  - drowsy
  - yawn
  - distraction
  - no_face
"""

import argparse
import csv
import os
import time
from datetime import datetime

import cv2


LABEL_KEYS = {
    ord("1"): "alert",
    ord("2"): "drowsy",
    ord("3"): "yawn",
    ord("4"): "distraction",
    ord("5"): "no_face",
}


def ensure_layout(root_dir):
    os.makedirs(root_dir, exist_ok=True)
    clips_dir = os.path.join(root_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    for label in LABEL_KEYS.values():
        os.makedirs(os.path.join(clips_dir, label), exist_ok=True)
    return clips_dir


def append_row(csv_path, row):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "clip_id",
                "label",
                "file_path",
                "fps",
                "frame_count",
                "duration_sec",
                "recorded_at",
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def start_writer(path, fps, width, height):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (width, height))


def main():
    parser = argparse.ArgumentParser(description="Collect labeled driving clips from webcam.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--output", default="data/v1", help="Dataset root folder (default: data/v1)")
    parser.add_argument("--fps", type=float, default=20.0, help="Recording FPS (default: 20)")
    args = parser.parse_args()

    clips_dir = ensure_layout(args.output)
    labels_csv = os.path.join(args.output, "labels.csv")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera permissions/index.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

    active_writer = None
    active_label = None
    active_path = None
    active_started = None
    frame_count = 0

    print("Dataset collection started")
    print("Press 1=alert, 2=drowsy, 3=yawn, 4=distraction, 5=no_face")
    print("Press R to stop current recording, Q to quit")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame read failed, stopping.")
                break

            overlay = f"Recording: {active_label if active_label else 'none'}"
            cv2.putText(frame, overlay, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 220, 30), 2)
            cv2.putText(
                frame,
                "1:alert 2:drowsy 3:yawn 4:distraction 5:no_face R:stop Q:quit",
                (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            cv2.imshow("Dataset Collector", frame)

            if active_writer is not None:
                active_writer.write(frame)
                frame_count += 1

            key = cv2.waitKey(1) & 0xFF

            # Start recording for selected label
            if key in LABEL_KEYS:
                # If recording already active, finalize it first.
                if active_writer is not None:
                    active_writer.release()
                    duration = max(0.0, time.time() - active_started)
                    append_row(
                        labels_csv,
                        {
                            "clip_id": os.path.splitext(os.path.basename(active_path))[0],
                            "label": active_label,
                            "file_path": active_path,
                            "fps": args.fps,
                            "frame_count": frame_count,
                            "duration_sec": round(duration, 3),
                            "recorded_at": datetime.utcnow().isoformat(),
                        },
                    )

                active_label = LABEL_KEYS[key]
                stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                clip_id = f"{active_label}_{stamp}"
                active_path = os.path.join(clips_dir, active_label, f"{clip_id}.mp4")
                active_writer = start_writer(active_path, args.fps, width, height)
                active_started = time.time()
                frame_count = 0
                print(f"Recording started: {active_path}")

            # Stop current recording
            elif key in (ord("r"), ord("R")) and active_writer is not None:
                active_writer.release()
                duration = max(0.0, time.time() - active_started)
                append_row(
                    labels_csv,
                    {
                        "clip_id": os.path.splitext(os.path.basename(active_path))[0],
                        "label": active_label,
                        "file_path": active_path,
                        "fps": args.fps,
                        "frame_count": frame_count,
                        "duration_sec": round(duration, 3),
                        "recorded_at": datetime.utcnow().isoformat(),
                    },
                )
                print(f"Recording saved: {active_path}")
                active_writer = None
                active_label = None
                active_path = None
                active_started = None
                frame_count = 0

            elif key in (ord("q"), ord("Q")):
                break

    finally:
        if active_writer is not None:
            active_writer.release()
            duration = max(0.0, time.time() - active_started)
            append_row(
                labels_csv,
                {
                    "clip_id": os.path.splitext(os.path.basename(active_path))[0],
                    "label": active_label,
                    "file_path": active_path,
                    "fps": args.fps,
                    "frame_count": frame_count,
                    "duration_sec": round(duration, 3),
                    "recorded_at": datetime.utcnow().isoformat(),
                },
            )
            print(f"Recording saved before exit: {active_path}")

        cap.release()
        cv2.destroyAllWindows()
        print(f"Done. Labels file: {labels_csv}")


if __name__ == "__main__":
    main()
