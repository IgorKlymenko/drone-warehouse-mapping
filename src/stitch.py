import cv2
import os
import sys
from natsort import natsorted  # sorting purposes


import imutils
from aruco_config import ARUCO_DICT

cv2.ocl.setUseOpenCL(False)

# Parser Arguments for code dev
#ap = argparse.ArgumentParser()
# Remove argparse related code
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image containing ArUCo tag")
# ap.add_argument("-t", "--type", type=str,
# 	default="DICT_ARUCO_ORIGINAL",
# 	help="type of ArUCo tag to detect")
# args = vars(ap.parse_args())

# Stitcher confidence
# CONFIDENCE = 0.1

arucoParams = cv2.aruco.DetectorParameters()


# Adjust detection parameters to decrease the detection threshold


arucoParams.cornerRefinementMinAccuracy = 0.01
arucoParams.adaptiveThreshWinSizeMin = 5
arucoParams.adaptiveThreshWinSizeMax = 23
arucoParams.adaptiveThreshWinSizeStep = 10
arucoParams.minMarkerPerimeterRate = 0.05
arucoParams.maxMarkerPerimeterRate = 6.0

# Add these parameters to further refine detection
arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
arucoParams.cornerRefinementWinSize = 5
arucoParams.cornerRefinementMaxIterations = 30
arucoParams.cornerRefinementMinAccuracy = 0.01

# Adjust thresholding
arucoParams.minOtsuStdDev = 5.0
arucoParams.perspectiveRemovePixelPerCell = 8
arucoParams.perspectiveRemoveIgnoredMarginPerCell = 0.13

def img_stitch(IMAGE_DIR):  
    """Actual image stitching"""
    stitcher = cv2.Stitcher_create()

    # Extract all frames from the directory
    frames = []
    for filename in natsorted(os.listdir(IMAGE_DIR)):
        if filename.endswith(".jpg"):
            frame_path = os.path.join(IMAGE_DIR, filename)
            frame = cv2.imread(frame_path)
            if frame is not None:
                frames.append(frame)
            else:
                print(f"Warning: Could not read frame {filename}")

    # Edge case
    if len(frames) == 0:
        print("Error: No frames to stitch.")
        return

    #stitcher.setPanoConfidenceThresh(CONFIDENCE)  # Set the threshold value as needed
    status, stitched_image = stitcher.stitch(frames)

    # Debugging
    if status != cv2.Stitcher_OK:
        print("Error during stitching. Status code:", status)
        if status == cv2.Stitcher_ERR_NEED_MORE_IMGS:
            print("Error: Need more images to perform stitching.")
        elif status == cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL:
            print("Error: Homography estimation failed.")
        elif status == cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL:
            print("Error: Camera parameters adjustment failed.")
        else:
            print("Error: Unknown error occurred during stitching.")

    return stitched_image

def set_stitch(IMAGE_DIR):
    """Function that calls img_stitch on the folder with images to stitch"""
    for filename in natsorted(os.listdir(IMAGE_DIR)):
        # only the per-bay frame folders, never our own output folders
        if filename.startswith("set_") and os.path.isdir(os.path.join(IMAGE_DIR, filename)):
            stitched_image = img_stitch(os.path.join(IMAGE_DIR, filename))
            
            if not os.path.exists(os.path.join(IMAGE_DIR, "stitched")):
                os.makedirs(os.path.join(IMAGE_DIR, "stitched"))

            if stitched_image is not None:
                cv2.imwrite(os.path.join(IMAGE_DIR, "stitched", f"{filename}.jpg"), stitched_image)
            else:
                print(f"Skipping {filename} due to stitching error.")

def mark_aruco(SOURCE_DIR, IMAGE_DIR, aruco_type="DICT_4X4_1000"):
    """Detect Aruco + Measures Detects Corner and Central parts of the Aruco"""
    if not os.path.exists(os.path.join(IMAGE_DIR, "stitched_aruco")):
        os.makedirs(os.path.join(IMAGE_DIR, "stitched_aruco"))


    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Directory {SOURCE_DIR} does not exist.")
        return

    for filename in natsorted(os.listdir(SOURCE_DIR)):
        if not filename.endswith(".jpg"):
            continue
        image_path = os.path.join(SOURCE_DIR, filename)
        print("[INFO] loading image...")
        image = cv2.imread(image_path)
        image = imutils.resize(image, width=4900) #used sample width based on stitching 5 frames per pano


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

                # draw the bounding box of the ArUCo detection
                cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
                cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
                cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
                cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)

                # compute and draw the center (x, y)-coordinates of the ArUco
                # marker
                cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                cY = int((topLeft[1] + bottomRight[1]) / 2.0)

                
                # mathematics for line
                ### function to find slope 
                def slope(p1,p2):
                    x1,y1=p1
                    x2,y2=p2
                    if x2!=x1:
                        return((y2-y1)/(x2-x1))
                    else:
                        return 'NA'

                ### main function to draw lines between two points
                def drawLine(image,p1,p2):
                    x1,y1=p1
                    x2,y2=p2
                    ### finding slope
                    m=slope(p1,p2)
                    ### getting image shape
                    h,w=image.shape[:2]

                    if m!='NA':
                        ### here we are essentially extending the line to x=0 and x=width
                        ### and calculating the y associated with it
                        ##starting point
                        px=0
                        py=-(x1-0)*m+y1
                        ##ending point
                        qx=w
                        qy=-(x2-w)*m+y2
                    else:
                    ### if slope is zero, draw a line with x=x1 and y=0 and y=height
                        px,py=x1,0
                        qx,qy=x1,h
                    cv2.line(image, (int(px), int(py)), (int(qx), int(qy)), (0, 0, 255), 6)


                dot1 = (cX, cY)
                y_shift = topRight[1] - bottomRight[1]
                x_shift = bottomRight[0] - topRight[0]

                #dot2 = (cX - x_shift, cY + y_shift)
                dot2 = (cX, cY + 10)
                print(topRight, bottomRight, dot1, dot2)
                drawLine(image, dot1, dot2)

                cv2.circle(image, (cX, cY), 100, (0, 0, 255), 8)

                # draw the ArUco marker ID on the image
                cv2.putText(image, str(markerID),
                    (topLeft[0] - 40, topLeft[1] - 40), cv2.FONT_HERSHEY_SIMPLEX,
                    3, (0, 255, 255), 10, 2)
                #cv2.line(image, dot1, dot2,(255, 0, 0), 2)
                print("[INFO] ArUco marker ID: {}".format(markerID))

            # upload the output image
            cv2.imwrite(os.path.join(IMAGE_DIR, "stitched_aruco", f"{filename}"), image)
            if image is None:
                print(f"Skipping {filename} due to stitching error.")
