from ultralytics import YOLO
from natsort import natsorted
import os
import urllib.request
import torch
from segment_anything import sam_model_registry, SamPredictor
import cv2
import numpy as np
from scipy.ndimage import zoom

from matplotlib import pyplot as plt
import yaml

from sklearn.linear_model import LinearRegression
from scipy.ndimage import rotate

import torch



def rotate_yolo(BASE_DIR, IMAGE_DIR):
    WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
    # Download pretrained file for Segment Anything Model (SAM)
    if "sam_vit_h_4b8939.pth" not in os.listdir(os.path.join(WEIGHTS_DIR, "sam")):
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        filename = os.path.join(WEIGHTS_DIR, "sam", "sam_vit_h_4b8939.pth")
        urllib.request.urlretrieve(url, filename)
        print(f"Downloaded {filename}")

    # Not adviced to run it on the CPU - RTX 3070 Ti laptop was sufficient to complete analysis in 130ms/image (480x1024)
    # Checks CUDA to run on GPU, runs on the CPU otherwise
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device being used: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")

    # Running on CPU / GPU
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(DEVICE)

    # SAM model used
    MODEL_TYPE = "vit_h"

    # weights for the SAM model - downloadeed at the beggining
    CHECKPOINT_PATH = os.path.join(WEIGHTS_DIR, "sam/sam_vit_h_4b8939.pth")

    # SAM model settigns for SEGMENTATION
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    predictor = SamPredictor(sam)

    # YOLO model settigns for object IDENTIFICATION 
    YOLOmodel = YOLO(os.path.join(WEIGHTS_DIR, "yolo/vert-stitch-YOLOv9.pt"))

    STITCHED_DIR = os.path.join(IMAGE_DIR, "stitched")

    # For each image in the file directory 
    for filename in natsorted(os.listdir(STITCHED_DIR)):
        if filename.endswith(".jpg"):

            # Object Detection using YOLO
            image_path = os.path.join(STITCHED_DIR, filename)
            results = YOLOmodel.predict(image_path, imgsz = 1024, conf = 0.1, save = True, save_txt = True, save_conf = True)   ## 0.02 is the most optimmal for this 30 images trained model

            name, ext = os.path.splitext(filename)
            new_filename = name + '.txt'

            YOLO_DIR = os.path.join('runs', 'detect', natsorted(os.listdir(os.path.join('runs', 'detect')))[-1])
            PROC_IMG_DIR = os.path.join(YOLO_DIR, "labels", new_filename)

            # Going into the basic new dir of YOLO output results
            processed_yolo = os.path.join(YOLO_DIR, natsorted(os.listdir(STITCHED_DIR))[-1])
            print(processed_yolo) # Post-Porocessed RESULTS 1
            # Read the image to get its dimensions
            image = cv2.imread(image_path)
            image_height, image_width, _ = image.shape


            # Piece of code that can segment evetything on the image - 11 G of RAM required for 480 x 1024 image
            # My RTX 3070 Ti Laptop was not able to complete it solo. Worked slowly, but surely combined with CPU
            """mask_generator = SamAutomaticMaskGenerator(sam)
            image_bgr = cv2.imread(os.path.join(processed_yolo, filename))
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            torch.cuda.empty_cache()

            # Enable mixed precision
            scaler = torch.cuda.amp.GradScaler()

            # In your main processing loop
            with torch.cuda.amp.autocast():
                sam_result = mask_generator.generate(image_rgb)

            # Delete variables you no longer need
            del sam_result
            torch.cuda.empty_cache()
            """
            
            did_first = False
            difference = 0
            # Reading the TXT boundaries output by YOLO model
            bboxes = []
            class_ids = []
            conf_scores = []

            with open(PROC_IMG_DIR, 'r') as file:
                for line in file:
                    components = line.split()
                    class_id = int(components[0])
                    confidence = float(components[5])
                    cx, cy, w, h = [float(x) for x in components[1:5]]

                    # Convert from normalized [0, 1] to image scale
                    cx *= image_width
                    cy *= image_height
                    w *= image_width
                    h *= image_height

                    # Convert the center x, y, width, and height to xmin, ymin, xmax, ymax
                    xmin = cx - w / 2
                    ymin = cy - h / 2
                    xmax = cx + w / 2
                    ymax = cy + h / 2

                    class_ids.append(class_id)
                    bboxes.append((xmin, ymin, xmax, ymax))
                    conf_scores.append(confidence)


            # Display the results from the YOLO OBJ DETECTION
            for class_id, bbox, conf in zip(class_ids, bboxes, conf_scores):
                print(f'Class ID: {class_id}, Confidence: {conf:.2f}, BBox coordinates: {bbox}')


            # Opennign image into cv2 and setttign it to the SAM predictor
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            predictor.set_image(image)


            # Using YAML file for segmentation classes 
            with open(os.path.join(WEIGHTS_DIR, "configs/data.yaml"), 'r') as file:
                coco_data = yaml.safe_load(file)
                class_names = coco_data['names']

            color_map = {}
            for class_id in class_ids:
                color_map[class_id] = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)

            # Adding the mask of segmented pixels into axes on the image
            def show_mask(mask, ax, color):
                h, w = mask.shape[-2:]
                mask_image = mask.reshape(h, w, 1) * np.array(color).reshape(1, 1, -1)
                ax.imshow(mask_image)

            # Adding the box of segmented pixels into axes on the image
            def show_box(box, label, conf_score, color, ax):
                x0, y0 = box[0], box[1]
                w, h = box[2] - box[0], box[3] - box[1]
                rect = plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor='none', lw=2)
                ax.add_patch(rect)

                label_offset = 10

                # Construct the label with the class name and confidence score
                label_text = f'{label} {conf_score:.2f}'

                ax.text(x0, y0 - label_offset, label_text, color='black', fontsize=10, va='top', ha='left',
                        bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', boxstyle='square,pad=0.4'))

            plt.figure(figsize=(10, 10))
            ax = plt.gca()
            plt.imshow(image)

            y_coords = []
            x_coords = []
            

            for class_id, bbox in zip(class_ids, bboxes):
                class_name = class_names[class_id]
                # print(f'Class ID: {class_id}, Class Name: {class_name}, BBox coordinates: {bbox}')

                color = color_map[class_id]
                input_box = np.array(bbox)

                # Generate the mask for the current bounding box
                # THE MOST COMPUTATION CONSUMING PART - SAM model is doing segmentation
                masks, _, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box,
                    multimask_output=False,
                )

                # Work with masks
                mask = masks[0]

                # Get the coordinates of the mask where it is True
                mask_coords = np.argwhere(mask)

                # Generate a trendline for the further alignment
                if mask_coords.size > 0:
                    y_coords.append(mask_coords[:, 0]) 
                    x_coords.append(mask_coords[:, 1])


                y_coords_flat = np.concatenate(y_coords)
                x_coords_flat = np.concatenate(x_coords)

                # Attempt to do image scaling - works somehow, but does not have any practical use case
                """
                if not did_first:
                    min_x = np.min(x_coords_flat)
                    max_x = np.max(x_coords_flat)
                    avg_x = (min_x + max_x) / 2

                    # Calculate the y-coordinates corresponding to min_x and max_x
                    y_min_x = np.min(y_coords_flat[x_coords_flat == min_x].mean())
                    y_max_x = np.max(y_coords_flat[x_coords_flat == max_x].mean())

                    # Calculate the difference between the minimum y and maximum y
                    difference = y_max_x - y_min_x
                    did_first = True


                min_x = np.min(x_coords_flat)
                max_x = np.max(x_coords_flat)
                avg_x = (min_x + max_x) // 2

                # Calculate the y-coordinates
                y_min_x = np.min(y_coords_flat[x_coords_flat == min_x].mean())
                y_max_x = np.max(y_coords_flat[x_coords_flat == max_x].mean())

                # Calculate the difference between the minimum y and maximum y
                y_difference = y_max_x - y_min_x
                scaling_factor = difference / y_difference
            
                # Scale the image
                scaled_image = zoom(image, (scaling_factor, scaling_factor, 1), order=1)
                """

                # Calculate the average y-coordinate for each unique x-coordinate in the FRAME
                unique_x_coords = np.unique(x_coords_flat)
                average_y_coords = np.array([y_coords_flat[x_coords_flat == x].mean() for x in unique_x_coords])

                # Plot the line connecting these average points
                ax.plot(unique_x_coords, average_y_coords, color='orange', lw=1)

                # Fit a linear regression model to these points to get the TRENDLINE for the WHOLE FRAME
                model = LinearRegression()
                model.fit(unique_x_coords.reshape(-1, 1), average_y_coords)
                trendline_y = model.predict(unique_x_coords.reshape(-1, 1))

                # Plot the trendline
                ax.plot(unique_x_coords, trendline_y, color='red', lw=1, linestyle='--')

                # Calculate the angle of the trendline
                angle = np.arctan(model.coef_[0]) * (180 / np.pi)

                # Rotate the image
                rotated_image = rotate(image, angle, reshape=True)

                show_mask(mask, ax, color=color)
                #show_box(bbox, class_name, conf, color, ax)

            # Show the final plot
            plt.figure(figsize=(10, 10))
            plt.imshow(rotated_image)
            plt.axis('off')

            if "rotated" not in os.listdir(IMAGE_DIR):
                os.mkdir(os.path.join(IMAGE_DIR, "rotated"))

            plt.savefig(os.path.join(IMAGE_DIR, "rotated", filename), bbox_inches='tight', pad_inches=0, dpi=300)

            plt.close('all')
            print(f"Saved {filename[:-4]} after the segmentation and proper roatation into the directory ../rotated/{filename}")
            #plt.show()