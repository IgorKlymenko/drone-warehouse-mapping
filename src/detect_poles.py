import os
import sys

import cv2
import torch
from ultralytics import YOLO
from natsort import natsorted

import argparse
import imutils
import shutil
from aruco_config import ARUCO_DICT


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DIR = os.path.join(BASE_DIR, "data", "raw")
FRAME_DIR = os.path.join(BASE_DIR, "data", "frames", "raw")
CLAS_DIR = os.path.join(BASE_DIR, "data", "frames", "detected")
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
MARK_DIR = os.path.join(BASE_DIR, "data", "frames", "marked")
FOLD_DIR = os.path.join(BASE_DIR, "data")

VIDEO_DIR = os.path.join(BASE_DIR, "data/raw", "sample1_long.mp4")



cv2.ocl.setUseOpenCL(False)

ap = argparse.ArgumentParser()
# Remove argparse related code
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image containing ArUCo tag")
# ap.add_argument("-t", "--type", type=str,
# 	default="DICT_ARUCO_ORIGINAL",
# 	help="type of ArUCo tag to detect")
# args = vars(ap.parse_args())

CONFIDENCE = 0.3
IMG_SCALE = 1024 # MODEL WAS TRAINED ON 1024
FRAMES_PER_PANO = 6
FRAMES_PER_IMAGE = 8
arucoParams = cv2.aruco.DetectorParameters()


# Prior Using - Adapt parameters specific warehouse operations aer conducted in

arucoParams.adaptiveThreshWinSizeMin = 4
arucoParams.adaptiveThreshWinSizeMax = 23
arucoParams.adaptiveThreshWinSizeStep = 10
arucoParams.minMarkerPerimeterRate = 0.1
arucoParams.maxMarkerPerimeterRate = 6.0
arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
arucoParams.cornerRefinementWinSize = 5
arucoParams.cornerRefinementMaxIterations = 10
arucoParams.cornerRefinementMinAccuracy = 0.1
arucoParams.minOtsuStdDev = 5.0
arucoParams.perspectiveRemovePixelPerCell = 8
arucoParams.perspectiveRemoveIgnoredMarginPerCell = 0.13


class Pole:
    def __init__(self, id, start, end, lifetime) -> None:
        self.id = id
        self.start = start
        self.end = end
        self.lifetime = lifetime


def extract_frames(SOURCE_DIR, FRAME_DIR, FRAME_RATE, rotate=False):
    os.makedirs(FRAME_DIR, exist_ok=True)

    # Process each video in the source directory
    video_path = os.path.join(SOURCE_DIR)
    cap = cv2.VideoCapture(video_path)

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        # Process every 10th frame
        if frame_id % FRAME_RATE == 0:
            if rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            output_path = os.path.join(FRAME_DIR, f"frame_{frame_id//FRAME_RATE:04d}.jpg")
            cv2.imwrite(output_path, frame)

    cap.release()

    print(f"Frame extraction completed. Frames saved in {FRAME_DIR}.")


def detect_poles(FRAME_DIR, OUTPUT_DIR, WEIGHTS_DIR, POLES_CONF, yolo_weights=None, aruco_type="DICT_4X4_1000"):
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    poles = {}


    
    # Check CUDA availability
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device being used: {DEVICE}")

    # Load YOLO model
    YOLOmodel = YOLO(yolo_weights or os.path.join(WEIGHTS_DIR, "yolo", "vert-stitch-YOLOv9.pt"))

    DETECTED_DIR = OUTPUT_DIR
    os.makedirs(DETECTED_DIR, exist_ok=True)

    # Process each image in the frame directory
    for filename in natsorted(os.listdir(FRAME_DIR)):
        if filename.endswith(".jpg"):
            print("\n \n \nNEW IMAGE")
            image_path = os.path.join(FRAME_DIR, filename)

            #DETECT ARUCOS

            aruco_markers, aruco_width = detect_center_aruco(image_path, resolution= 1024, aruco_type=aruco_type)


            frame = cv2.imread(image_path)
            frame = imutils.resize(frame, width = IMG_SCALE)

            # Object detection using YOLO
            results = YOLOmodel(frame, imgsz = IMG_SCALE, conf=POLES_CONF)

            for result in results:
                boxes = result.boxes  # Access boxes directly from the result object
                for box in boxes:
                    # Extract bounding box coordinates
                    xyxy = box.xyxy[0].cpu().numpy()  # Get the box coordinates
                    xmin, ymin, xmax, ymax = map(int, xyxy)

                    confidence = float(box.conf)
                    det = box.cls

                    if det == 1:
                        label = f'Beam - {confidence:.2f}'
                    elif det == 0:
                        label = f'Pole - {confidence:.2f}'
                        # Draw bounding box on the frame
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
                        cv2.putText(frame, label, (xmin, ymin + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)

                        width = xmax - xmin # kind of std
                        for tag in aruco_markers: # for every detected tag
                            #print(f"tag: {tag}, xCentral: {aruco_markers.get(tag)[1]}")
                            print(f"tag: {tag}, centrX: {aruco_markers.get(tag)[0]}, centrY: {aruco_markers.get(tag)[1]}, min {xmin- (width * 3)}, max {xmax + (width * 3)}") # manual confidence interval 
                            if aruco_markers.get(tag)[0] > xmin- (width * 3) and aruco_markers.get(tag)[0] < xmax + (width * 3): # tag belongs to the pole
                                if tag not in poles:
                                    poles[tag] = Pole(id = tag, start = filename, end = filename, lifetime=0)
                                else:
                                    pole = poles[tag]
                                    pole.end = filename
                                    pole.lifetime += 1

            # Save the frame with detected objects
            output_path = os.path.join(DETECTED_DIR, filename)
            cv2.imwrite(output_path, frame)

    print(f"Detection completed. Frames saved in {DETECTED_DIR}.")
    return poles




def detect_center_aruco(image_path, resolution, aruco_type="DICT_4X4_1000"):
    """Helper Function - Detect Aruco + Measures Detects Corner and Central parts of the Aruco"""

    aruco_center = {}
    aruco_width = {}
    
    print("[INFO] loading image...")
    image = cv2.imread(image_path)
    image = imutils.resize(image, width = resolution)

    args = {
        "image": str(image_path),  # Set your image path here
        "type": aruco_type       # Set the ArUCo tag type here
    }

        
    if ARUCO_DICT.get(args["type"], None) is None:
        print("[INFO] ArUCo tag of '{}' is not supported".format(
            args["type"]))
        sys.exit(0)

    # load the ArUCo dictionary, grab the ArUCo parameters, and detect
    # the markers
    print("[INFO] detecting '{}' tags...".format(args["type"]))
    arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[args["type"]])

    detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)
    (corners, ids, rejected) = detector.detectMarkers(image)
    # verify *at least* one ArUco marker was detected
    if len(corners) > 0:
        # flatten the ArUco IDs list
        ids = ids.flatten()

        # loop over the detected ArUCo corners
        for (markerCorner, markerID) in zip(corners, ids):
            # extract the marker corners (which are always returned in
            # top-left, top-right, bottom-right, and bottom-left order)
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            # convert each of the (x, y)-coordinate pairs to integers
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            # compute and draw the center (x, y)-coordinates of the ArUco marker
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)

            width = abs(int(bottomLeft[0]) - int(bottomRight[0]))
            height = abs(int(topLeft[1]) - int(bottomLeft[1]))

            if width == 0:
                width = 1

            if height == 0:
                height = 1
            print(width, height)

            hor_stratch  = width / height

            print(f"hor {hor_stratch}")

            aruco_width[markerID] = (width * hor_stratch + height / hor_stratch) / 2.0


            #cv2.line(image, dot1, dot2,(255, 0, 0), 2)

            aruco_center[markerID] = (cX, cY)
            print("[INFO] ArUco marker ID: {}".format(markerID))

        if image is None:
            print(f"Skipping {image_path} due to stitching error.")

    return aruco_center, aruco_width

def sort_images(SOURCE_DIR, DEST_DIR, FRAMES_PER_PANO, poles):
    print(f"Starting sort_images function")
    print(f"SOURCE_DIR: {SOURCE_DIR}")
    print(f"DEST_DIR: {DEST_DIR}")
    print(f"FRAMES_PER_PANO: {FRAMES_PER_PANO}")
    print(f"Number of poles: {len(poles)}")

    # Create destination directory if it doesn't exist
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)
        print(f"Created destination directory: {DEST_DIR}")

    pano_count = 0
    half_frames = FRAMES_PER_PANO // 2
    pole_ids = list(poles.keys())
    print(f"Pole IDs: {pole_ids}")

    for i in range(len(pole_ids) - 1):
        current_pole = poles[pole_ids[i]]
        next_pole = poles[pole_ids[i + 1]]
        print(f"\nProcessing poles: {current_pole.id} and {next_pole.id}")

        print(f"Searching for frames in range: {current_pole.start} to {current_pole.end}")
        current_frames = natsorted([f for f in os.listdir(SOURCE_DIR) if current_pole.start <= f <= current_pole.end])
        print(f"Found {len(current_frames)} frames for current pole")

        print(f"Searching for frames in range: {next_pole.start} to {next_pole.end}")
        next_frames = natsorted([f for f in os.listdir(SOURCE_DIR) if next_pole.start <= f <= next_pole.end])
        print(f"Found {len(next_frames)} frames for next pole")

        # Take last half of current pole frames and first half of next pole frames
        if FRAMES_PER_PANO % 2 == 0:
            panorama_frames = current_frames[-half_frames:] + next_frames[:half_frames]
        else:
            panorama_frames = current_frames[-half_frames:] + next_frames[:half_frames+1]
        print(f"Selected {len(panorama_frames)} frames for panorama")

        if len(panorama_frames) == FRAMES_PER_PANO:
            pano_dir = os.path.join(DEST_DIR, f"set_{pano_count}")
            os.makedirs(pano_dir, exist_ok=True)
            print(f"Created panorama directory: {pano_dir}")

            for frame in panorama_frames:
                src_path = os.path.join(SOURCE_DIR, frame)
                dst_path = os.path.join(pano_dir, frame)
                shutil.copy(src_path, dst_path)
                print(f"Copied {frame} to {pano_dir}")

            pano_count += 1
            print(f"Completed panorama {pano_count}")
        else:
            print(f"Skipped panorama creation: insufficient frames ({len(panorama_frames)} != {FRAMES_PER_PANO})")

    print(f"\nFinished processing. Created {pano_count} panorama sets.")
