# Stitching drone footage into measured warehouse aisles

Turns a drone flyover video of warehouse racking into stitched panoramas and measures the distance between rack poles from the ArUco tags mounted on them. For mapping rack layouts without walking the aisles with a tape measure.

![whole aisle stitched from drone video](docs/aisle.jpg)

<sub>One aisle from a single drone pass, stitched bay by bay. Shown as two rows.</sub>

## Why I built this

Measuring pole-to-pole distances along warehouse racking by hand is slow and error-prone. The racks already carried ArUco tags of known size on every pole, and a drone could fly an aisle in minutes. So I wrote a pipeline that rebuilds each rack bay from the video and reads the distances off the tags.

## How it works

End to end: raw drone frames in, every bay stitched and measured, bays accumulated into the whole aisle.

![end to end: drone frames to pole-to-pole distances](docs/pipeline_v3.gif)

1. **Pick frames around each pole.** `src/detect_poles.py` runs YOLO on every extracted frame, tracks poles by their ArUco tag, and groups frames so each set spans one pole-to-pole gap.
2. **Stitch each set.** `src/stitch.py` hands the set to OpenCV's Stitcher.
3. **Level the panorama.** `src/segm_rotate.py` detects the shelf beam with YOLO, masks it with SAM, fits a line through the mask, and rotates by that angle.
4. **Measure.** `src/measure_aruco.py` detects the tags, derives a pixels-to-metric ratio from the known tag size, and reports the distance between tag centres.
5. **Repeat per bay.** Steps 1 to 4 run for every pole-to-pole gap along the aisle, and the stitched bays accumulate into the full-aisle panorama at the top of this page.

## Results

The whole aisle as one panorama at full resolution. Click to zoom:

![full aisle panorama at native resolution](docs/aisle_panorama.jpg)

Four bays from one run, tags detected and boxed:

![four stitched bays with detected tags](docs/bays.jpg)

The last step prints every pole-to-pole distance (in meters) and plots their distribution across the aisle:

<img src="docs/distances.png" width="520" alt="histogram of pole-to-pole distances">

## Run it locally

```bash
git clone https://github.com/IgorKlymenko/drone-warehouse-mapping.git
cd drone-warehouse-mapping
uv run src/main.py --video path/to/flight.mp4        # uv installs the dependencies itself
# or: pip install -r requirements.txt && python src/main.py --video path/to/flight.mp4
```

Two model files are needed. SAM downloads itself on first use. The YOLO pole/beam model is custom-trained and available on request (open an issue or email me); put it at `weights/yolo/vert-stitch-YOLOv9.pt` or pass `--yolo-weights`. The script checks for both before it starts and tells you what's missing.

Everything lands in `output/`: per-bay frame sets, `stitched/`, `rotated/`, `stitched_aruco/`, then `distances.csv` and `distances.png`. Thresholds, tag sizes in metres, the ArUco dictionary and all paths are flags; `--help` lists them. The run is split into seven stages and you can resume or run a slice:

```bash
uv run src/main.py --video flight.mp4 --stop-after stitch        # stop before the GPU steps
uv run src/main.py --video flight.mp4 --start-at rotate          # pick up from existing output/
uv run src/main.py --video flight.mp4 --poles-conf 0.5 --tag-size-even 0.12 --no-show
```

A GPU helps for the `detect` and `rotate` stages. The `stitch`, `mark` and `measure` stages need only OpenCV.

## The hard part

Deciding which frames go into which panorama. The first version (June 2024) cut the video into fixed windows of `frame_rate * frames_per_pano` frames and stitched each window. That breaks whenever the drone changes speed: a panorama ends up with one pole, or three, and the distance measurement is meaningless.

The fix in `detect_poles.py` runs YOLO on every extracted frame, tracks each pole by the ArUco tag nearest to it, and records the first and last frame where that pole is visible. A panorama is then built from the last three frames of pole N plus the first three of pole N+1, so every stitched image spans exactly one gap and contains both tags it needs.

The tradeoff is a custom-trained YOLO model to maintain, a GPU-sized dependency for a step that used to be arithmetic, and a hard requirement that every pole carry a tag. I took that over fixed windows because the measurement is only valid if the panorama boundaries follow the poles.

## Stack

Python, OpenCV (Stitcher, ArUco), Ultralytics YOLO, Segment Anything, PyTorch, scikit-learn, NumPy, matplotlib.

## Status

Works end to end on the sample footage. Still missing:

- The YOLO weights are available on request rather than in the repo, so a fresh clone runs the OpenCV stages out of the box and the full pipeline once you have them.
