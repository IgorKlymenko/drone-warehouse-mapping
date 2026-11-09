# Step: Stitch the images together based on the ArUco markers
import cv2
import os
import numpy as np
import sys

from natsort import natsorted # pip install natsort # sorting purposes

import argparse
from aruco_config import ARUCO_DICT, ARUCO_METRIC

IMG_SCALE = 4900 # Video Scale of 4900 used for stitched images after the first cycle

ap = argparse.ArgumentParser()
# Remove argparse related code
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image containing ArUCo tag")
# ap.add_argument("-t", "--type", type=str,
# 	default="DICT_ARUCO_ORIGINAL",
# 	help="type of ArUCo tag to detect")
# args = vars(ap.parse_args())


# Adjust detection parameters to decrease the detection threshold
arucoParams = cv2.aruco.DetectorParameters()

arucoParams.adaptiveThreshWinSizeMin = 4
arucoParams.adaptiveThreshWinSizeMax = 25
arucoParams.adaptiveThreshWinSizeStep = 8
arucoParams.minMarkerPerimeterRate = 0.017
arucoParams.maxMarkerPerimeterRate = 4
arucoParams.cornerRefinementMinAccuracy = 0.01  

arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
arucoParams.cornerRefinementWinSize = 5
arucoParams.cornerRefinementMaxIterations = 10


def detect_aruco(IMAGE_DIR, aruco_type="DICT_4X4_1000"):
    if not os.path.exists(os.path.join(IMAGE_DIR, "rotated")):
        os.makedirs(os.path.join(IMAGE_DIR, "rotated"))

    stitched_dir = os.path.join(IMAGE_DIR, "stitched").replace("\\", "/")
    if not os.path.exists(stitched_dir):
        print(f"Error: Directory {stitched_dir} does not exist.")
        return

    # Dictionary to store the first and last detected ArUco markers for each image
    aruco_markers = {}
    aruco_width = {}

    for filename in natsorted(os.listdir(stitched_dir)):
        if filename.endswith(".jpg"):
            image_path = os.path.join(stitched_dir, filename)
            print("[INFO] loading image...")
            image = cv2.imread(image_path)

            image = cv2.resize(image, (IMG_SCALE, int(image.shape[0] * (IMG_SCALE / image.shape[1]))))  # used sample width based on stitching 5 frames per pano

            if ARUCO_DICT.get(aruco_type, None) is None:
                print("[INFO] ArUco tag of '{}' is not supported".format(aruco_type))
                sys.exit(0)

            # load the ArUco dictionary, grab the ArUco parameters, and detect the markers
            print("[INFO] detecting '{}' tags...".format(aruco_type))
            arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])



            detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)
            corners, ids, rejected = detector.detectMarkers(image)

            # verify *at least* one ArUco marker was detected
            if ids is not None:
                ids = ids.flatten()

                # Initialize dictionary entries for the first and last detected markers
                aruco_markers[filename + "_l"] = None
                aruco_markers[filename + "_r"] = None

                # loop over the detected ArUco corners
                for i, (markerCorner, markerID) in enumerate(zip(corners, ids)):
                    # extract the marker corners
                    corners = markerCorner.reshape((4, 2))
                    topLeft, topRight, bottomRight, bottomLeft = corners
                    topLeft = tuple(map(int, topLeft))
                    topRight = tuple(map(int, topRight))
                    bottomRight = tuple(map(int, bottomRight))
                    bottomLeft = tuple(map(int, bottomLeft))

                    # compute the center coordinates of the ArUco marker
                    cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                    cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                    center = (cX, cY)

                    width = abs(int(bottomLeft[0]) - int(bottomRight[0]))
                    height = abs(int(topLeft[1]) - int(bottomLeft[1]))

                    if width == 0:
                        width = 1

                    if height == 0:
                        height = 1
                    print(width, height)

                    hor_stratch  = width / height

                    print(f"hor {hor_stratch}")
                    aruco_width[filename + "_" + str(markerID)] = (width * hor_stratch + height / hor_stratch) / 2.0

                    # Draw the bounding box and ID on the image
                    cv2.polylines(image, [np.array([topLeft, topRight, bottomRight, bottomLeft], dtype=np.int32)], True, (0, 255, 0), 2)
                    cv2.putText(image, str(markerID), (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)



                    # Crazy binary choice soritng ahah
                    # 1 el: positioned as right
                    # 2 el: one is left another right
                    # 3 el there is a l, m, r
                    if aruco_markers.get(filename + "_r") is None:
                       aruco_markers[filename + "_r"] = [markerID, center]

                    elif center[0] > aruco_markers[filename + "_r"][1][0]:
                        
                        if aruco_markers[filename + "_l"] and aruco_markers[filename + "_r"][1][0] > aruco_markers[filename + "_l"][1][0]:
                            aruco_markers[filename + "_m"] = aruco_markers[filename + "_r"]
                        else:
                            aruco_markers[filename + "_l"] = aruco_markers[filename + "_r"]
                        aruco_markers[filename + "_r"] = [markerID, center]



                    elif aruco_markers.get(filename + "_l") is None:
                        aruco_markers[filename + "_l"] = [markerID, center]
                        
                    elif center[0] < aruco_markers[filename + "_l"][1][0]:

                        if aruco_markers[filename + "_r"] and aruco_markers[filename + "_l"][1][0] < aruco_markers[filename + "_r"][1][0]:
                            aruco_markers[filename + "_m"] = aruco_markers[filename + "_l"]
                        else:
                                aruco_markers[filename + "_r"] = aruco_markers[filename + "_l"]
                        aruco_markers[filename + "_l"] = [markerID, center]

                    else:
                        aruco_markers[filename + "_m"] = [markerID, center]
                        break
    return aruco_markers, aruco_width


def relative_distance(aruco_markers, aruco_width, IMAGE_DIR):
    rel_dist = {}

    for filename in natsorted(os.listdir(os.path.join(IMAGE_DIR, "stitched_aruco"))):
        print("[INFO] Processing file:", filename)
        if filename.endswith(".jpg"):
            if not aruco_markers.get(filename + '_l') or not aruco_markers.get(filename + '_r'):
                print(f"[WARN] {filename}: fewer than two tags found, skipping")
                continue
            origin = aruco_markers[filename + '_l'][0]
            if filename + '_m' in aruco_markers:
                dist1 = transform_into_metric(aruco_markers[filename + '_m'][1][0] - aruco_markers[filename + '_l'][1][0], origin, filename, aruco_width)
                dist2 = transform_into_metric(aruco_markers[filename + '_r'][1][0] - aruco_markers[filename + '_m'][1][0], origin, filename, aruco_width)
                rel_dist[f"{aruco_markers[filename + '_l'][0]}-{aruco_markers[filename + '_m'][0]}"] = dist1
                rel_dist[f"{aruco_markers[filename + '_m'][0]}-{aruco_markers[filename + '_r'][0]}"] = dist2
            else:
                dist1 = transform_into_metric(aruco_markers[filename + '_r'][1][0] - aruco_markers[filename + '_l'][1][0], origin, filename, aruco_width)
                rel_dist[f"{aruco_markers[filename + '_l'][0]}-{aruco_markers[filename + '_r'][0]}"] = dist1

    # Testing Display
    for key, value in rel_dist.items():
        print("Relative distance:", key, value)

    return rel_dist

def transform_into_metric(pix_dist, origin_tag, filename, aruco_width):
    orig_pix = aruco_width.get(filename + "_" + str(origin_tag))

    if origin_tag > 9 or (origin_tag < 930 and origin_tag > 943): # Included Edge Cases
        key = origin_tag % 2
    else:
        key = 99

    orig_metric = ARUCO_METRIC.get(str(key))
    ratio = orig_metric / orig_pix

    return pix_dist * ratio
