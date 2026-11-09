# Step: Stitch the images together based on the ArUco markers
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
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


def detect_aruco(IMAGE_DIR):
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

            aruco_type = "DICT_4X4_1000"
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


def stitch_frames_right_movement(aruco_markers, stitched_dir, IMAGE_DIR):
    sorted_files = natsorted(os.listdir(stitched_dir))

    def get_next_file(current_filename):
        try:
            current_index = sorted_files.index(current_filename)
            if filename.endswith(".jpg"): 
                return sorted_files[current_index + 1]  # Get the next file in the list
            else:
                return None
        except (ValueError, IndexError):
            return None
        
    for filename in sorted_files:
        if filename.endswith(".jpg"):

            # Open Image 1
            image1_path = os.path.join(stitched_dir, filename)
            print("[INFO] loading image...")
            image1 = cv2.imread(image1_path)
            image1 = cv2.resize(image1, (IMG_SCALE, int(image1.shape[0] * (IMG_SCALE / image1.shape[1]))))  # used sample width based on stitching 5 frames per pano


            # Open Image 2
            filename2 = get_next_file(filename)
            if filename2 is None:
                continue
            image2_path = os.path.join(stitched_dir, filename2)
            image2 = cv2.imread(image2_path)
            image2 = cv2.resize(image2, (IMG_SCALE, int(image2.shape[0] * (IMG_SCALE / image2.shape[1]))))  # used sample width based on stitching 5 frames per pano


            print(f"{filename} - {filename2}")
            
            # Get ArUco marker coordinates for alignment
            #mid1 = aruco_markers.get(filename + "_r")
            #mid2 = aruco_markers.get(filename2 + "_l")


            # MATCH tags from 2 consequent images
            ext = ["_r", "_m", "_l"]
            ext = ["_l", "_m", "_r"]

            found = False
            for e1 in ext:
                if not found:
                    mid1 = aruco_markers.get(filename + e1)   
                    for e2 in reversed(ext):
                        mid2 = aruco_markers.get(filename2 + e2)
                        if mid1 and mid2 and mid1[0] == mid2[0]:
                            found = True
                            break
                            

            # Create a blank canvas


            height = max(image1.shape[0], image2.shape[0])
            width = image1.shape[1] + image2.shape[1]
            canvas = np.zeros((height, width, 3), dtype=np.uint8)


            move_x = mid1[1][0] - mid2[1][0]
            move_y = mid1[1][1] - mid2[1][1]

            move_y = 0 # testing


            canvas_width = max(image1.shape[1] + abs(move_x), image2.shape[1] + abs(move_x))
            canvas_height = max(image1.shape[0] + abs(move_y), image2.shape[0] + abs(move_y))
            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
            print(canvas.shape)

            # Place image1 on the canvas
            start_y_image1 = max(0, -move_y)
            start_x_image1 = max(0, -move_x)

            canvas[start_y_image1:start_y_image1 + image1.shape[0], start_x_image1:start_x_image1 + image1.shape[1]] = image1

            # Calculate the starting x and y coordinates for image2 based on the move
            start_y_image2 = max(0, move_y)
            start_x_image2 = max(0, move_x)
            canvas[start_y_image2:start_y_image2 + image2.shape[0], start_x_image2:start_x_image2 + image2.shape[1]] = image2

            # Blend both images onto the canvas with alpha 0.5 and increase brightness
            alpha = 0.8
            beta = 10  # Increase brightness by adding a constant value

            # Blend image1 onto the canvas
            overlay1 = cv2.addWeighted(canvas[start_y_image1:start_y_image1 + image1.shape[0], :image1.shape[1]], 1 - alpha, image1, alpha, 0)
            canvas[start_y_image1:start_y_image1 + image1.shape[0], :image1.shape[1]] = cv2.convertScaleAbs(overlay1, alpha=1.0, beta=beta)

            # Blend image2 onto the canvas
            overlay2 = cv2.addWeighted(canvas[start_y_image2:start_y_image2 + image2.shape[0], start_x_image2:start_x_image2 + image2.shape[1]], 1 - alpha, image2, alpha, 0)
            canvas[start_y_image2:start_y_image2 + image2.shape[0], start_x_image2:start_x_image2 + image2.shape[1]] = cv2.convertScaleAbs(overlay2, alpha=1.0, beta=beta)

            print(f"{mid1} - {mid2}")

            if "stitched_aruco_t" not in os.listdir(IMAGE_DIR):
                os.mkdir(os.path.join(IMAGE_DIR, "stitched_aruco_t"))

            cv2.imwrite(os.path.join(IMAGE_DIR, "stitched_aruco_t", f"{filename}"), canvas)

           # plt.imshow(canvas)
            #plt.show()



def create_dag(aruco_markers, IMAGE_DIR):
    tags_sequence = []
    for filename in natsorted(os.listdir(os.path.join(IMAGE_DIR, "stitched_aruco"))):
        if filename.endswith(".jpg"):
            if filename + "_l" in aruco_markers and aruco_markers[filename + "_l"][0] not in tags_sequence:
                tags_sequence.append(aruco_markers[filename + "_l"][0])
            if filename + "_m" in aruco_markers and aruco_markers[filename + "_m"][0] not in tags_sequence:
                tags_sequence.append(aruco_markers[filename + "_m"][0])
            if filename + "_r" in aruco_markers and aruco_markers[filename + "_r"][0] not in tags_sequence:
                tags_sequence.append(aruco_markers[filename + "_r"][0])
    print(tags_sequence)
    return tags_sequence


def relative_distance(aruco_markers, aruco_width, IMAGE_DIR):
    rel_dist = {}

    for filename in natsorted(os.listdir(os.path.join(IMAGE_DIR, "stitched_aruco"))):
        print("[INFO] Processing file:", filename)
        if filename.endswith(".jpg"):
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
