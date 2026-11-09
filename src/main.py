"""Warehouse rack pipeline: drone video -> per-bay panoramas -> pole-to-pole distances.

Run from anywhere:
    python src/main.py --video data/raw/flight.mp4
    uv run src/main.py --video data/raw/flight.mp4

Every stage reads from and writes to --out. Use --start-at / --stop-after to resume or
run only part of the pipeline; outputs of earlier stages are reused.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import aruco_config  # noqa: E402
import detect_poles  # noqa: E402
import measure_aruco  # noqa: E402
import segm_rotate  # noqa: E402
import stitch  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGES = ["extract", "detect", "sort", "stitch", "rotate", "mark", "measure"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    io = p.add_argument_group("paths")
    io.add_argument("--video", required=True, help="drone video of one aisle")
    io.add_argument("--out", default=os.path.join(BASE_DIR, "output"), help="output folder (default: ./output)")
    io.add_argument("--weights-dir", default=os.path.join(BASE_DIR, "weights"), help="folder holding yolo/ and sam/ weights")
    io.add_argument("--yolo-weights", default=None, help="custom YOLO pole/beam model (default: <weights-dir>/yolo/vert-stitch-YOLOv9.pt)")
    io.add_argument("--rotate-video", action="store_true", help="rotate frames 90 degrees clockwise on extraction (landscape footage)")

    fr = p.add_argument_group("frame selection")
    fr.add_argument("--frame-rate", type=int, default=9, help="keep every Nth video frame (default 9)")
    fr.add_argument("--frames-per-pano", type=int, default=6, help="frames per bay panorama, even number (default 6)")

    th = p.add_argument_group("thresholds")
    th.add_argument("--poles-conf", type=float, default=0.6, help="YOLO confidence for pole detection when grouping frames (default 0.6)")
    th.add_argument("--beam-conf", type=float, default=0.1, help="YOLO confidence for beam detection when levelling (default 0.1)")

    tg = p.add_argument_group("ArUco tags")
    tg.add_argument("--aruco-dict", default="DICT_4X4_1000", choices=sorted(aruco_config.ARUCO_DICT), help="tag dictionary (default DICT_4X4_1000)")
    tg.add_argument("--tag-size-even", type=float, default=aruco_config.ARUCO_METRIC["0"], help="physical size of even-ID tags in metres (default %(default)s)")
    tg.add_argument("--tag-size-odd", type=float, default=aruco_config.ARUCO_METRIC["1"], help="physical size of odd-ID tags in metres (default %(default)s)")
    tg.add_argument("--tag-size-special", type=float, default=aruco_config.ARUCO_METRIC["99"], help="physical size of tags 1-9 and 931 in metres (default %(default)s)")

    ctl = p.add_argument_group("control")
    ctl.add_argument("--start-at", choices=STAGES, default=STAGES[0], help="first stage to run (reuse earlier outputs in --out)")
    ctl.add_argument("--stop-after", choices=STAGES, default=STAGES[-1], help="last stage to run")
    ctl.add_argument("--no-show", action="store_true", help="save the distance histogram without opening a window")
    return p.parse_args()


def check_inputs(a, stages):
    problems = []
    if "extract" in stages and not os.path.isfile(a.video):
        problems.append(f"video not found: {a.video}")
    yolo = a.yolo_weights or os.path.join(a.weights_dir, "yolo", "vert-stitch-YOLOv9.pt")
    if ("detect" in stages or "rotate" in stages) and not os.path.isfile(yolo):
        problems.append(f"YOLO weights not found: {yolo}\n  (custom pole/beam model, not in the repo; pass --yolo-weights or place it there)")
    if a.frames_per_pano % 2:
        problems.append("--frames-per-pano must be even")
    if problems:
        sys.exit("Cannot run:\n  " + "\n  ".join(problems))
    return yolo


def report(dist, out, show):
    if not dist:
        print("No pole-to-pole distances measured.")
        return
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vals = list(dist.values())
    print("\nPole-to-pole distances:")
    for k, v in dist.items():
        print(f"  {k}: {v:.4f}")
    print(f"\n{len(vals)} gaps  mean {np.mean(vals):.4f}  median {np.median(vals):.4f}  std {np.std(vals):.4f}")

    with open(os.path.join(out, "distances.csv"), "w") as f:
        f.write("tag_pair,distance_m\n")
        for k, v in dist.items():
            f.write(f"{k},{v}\n")

    plt.figure(figsize=(10, 6))
    plt.hist(vals, bins=min(20, max(5, len(vals))), alpha=0.6)
    plt.title("Pole-to-pole distances")
    plt.xlabel("distance (m)")
    plt.ylabel("count")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(os.path.join(out, "distances.png"), dpi=150, bbox_inches="tight")
    print(f"Saved {os.path.join(out, 'distances.csv')} and distances.png")
    if show:
        plt.show()


def main():
    a = parse_args()
    stages = STAGES[STAGES.index(a.start_at): STAGES.index(a.stop_after) + 1]
    if not stages:
        sys.exit("--start-at comes after --stop-after")
    yolo = check_inputs(a, stages)

    # Tag sizes live in one shared table that measure_aruco and detect_poles both read
    aruco_config.ARUCO_METRIC.update({"0": a.tag_size_even, "1": a.tag_size_odd, "99": a.tag_size_special})

    out = os.path.abspath(a.out)
    frame_dir = os.path.join(out, "sorting", "raw")
    detected_dir = os.path.join(out, "sorting", "detected")
    os.makedirs(out, exist_ok=True)
    print(f"Stages: {' -> '.join(stages)}\nOutput: {out}\n")

    poles = None
    if "extract" in stages:
        print("== extract: video -> frames")
        detect_poles.extract_frames(SOURCE_DIR=a.video, FRAME_DIR=frame_dir, FRAME_RATE=a.frame_rate, rotate=a.rotate_video)
    if "detect" in stages:
        print("== detect: YOLO poles + ArUco tags on every frame")
        poles = detect_poles.detect_poles(FRAME_DIR=frame_dir, OUTPUT_DIR=detected_dir, WEIGHTS_DIR=a.weights_dir,
                                          POLES_CONF=a.poles_conf, yolo_weights=yolo, aruco_type=a.aruco_dict)
    if "sort" in stages:
        if poles is None:
            sys.exit("'sort' needs the pole tracks from 'detect'; use --start-at detect (or earlier)")
        print("== sort: group frames into one set per pole-to-pole gap")
        detect_poles.sort_images(SOURCE_DIR=frame_dir, DEST_DIR=out, FRAMES_PER_PANO=a.frames_per_pano, poles=poles)
    if "stitch" in stages:
        print("== stitch: OpenCV Stitcher on every set_*/")
        stitch.set_stitch(out)
    if "rotate" in stages:
        print("== rotate: YOLO beam -> SAM mask -> line fit -> level")
        segm_rotate.rotate_yolo(WEIGHTS_DIR=a.weights_dir, IMAGE_DIR=out, yolo_weights=yolo, conf=a.beam_conf)
    if "mark" in stages:
        print("== mark: draw ArUco tags on the levelled panoramas")
        stitch.mark_aruco(SOURCE_DIR=os.path.join(out, "rotated"), IMAGE_DIR=out, aruco_type=a.aruco_dict)
    if "measure" in stages:
        print("== measure: tag size -> pixels per unit -> pole-to-pole distance")
        markers, widths = measure_aruco.detect_aruco(IMAGE_DIR=out, aruco_type=a.aruco_dict)
        dist = measure_aruco.relative_distance(markers, widths, out)
        report(dist, out, show=not a.no_show)
    print("\nDone.")


if __name__ == "__main__":
    main()
