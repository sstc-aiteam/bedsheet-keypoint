import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

realsense_data_folder = "/home/itrib40351/Documents/GitHub/bedSheetFoldingRobot/realsense/realsense_data2/"

def mask_to_polygon(image_file):
    # Read the masked image
    print(realsense_data_folder + "masked/" + image_file)

    if os.path.isfile(realsense_data_folder + "masked/" + image_file):
        masked_img = cv2.imread(realsense_data_folder + "masked/" + image_file)  # shape HxWx3

        # Create a mask where any nonzero pixel (in any channel) is considered foreground
        gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        image = cv2.imread(realsense_data_folder + "realsense_camera/" + image_file)                          # Original image

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print("No polygons found.")
        else:
            largest = max(contours, key=cv2.contourArea)
            # Optionally, simplify polygon (reduce vertices)
            epsilon = 0.002 * cv2.arcLength(largest, True)
            polygon = cv2.approxPolyDP(largest, epsilon, True)    # shape: (N,1,2)
            polygon = polygon.squeeze()                           # shape: (N,2)

            # # Draw the polygon on the image (green)
            # img_viz = image.copy()
            # cv2.polylines(img_viz, [polygon], isClosed=True, color=(0,255,0), thickness=2)
            # # Draw vertices (red dots)
            # for (x, y) in polygon:
            #     cv2.circle(img_viz, (int(x), int(y)), 5, (0,0,255), -1)

            # # Show result
            # plt.figure(figsize=(8,8))
            # plt.imshow(cv2.cvtColor(img_viz, cv2.COLOR_BGR2RGB))
            # plt.title('Polygon Vertices on Image')
            # plt.axis('off')
            # plt.show()
            # print("Polygon vertices:", polygon)

            # Define the mask's height and width (should match your image or ROI size)
            height = image.shape[0]
            width = image.shape[1]

            # Create an empty mask
            mask = np.zeros((height, width), dtype=np.uint8)

            # Draw/Fill the polygon on the mask; ensure shape is [1, N, 2]
            cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)

            # Save the binary mask
            cv2.imwrite(realsense_data_folder + "polygon_mask/" + image_file, mask)

for f in os.listdir(realsense_data_folder + "realsense_camera/"):
    if f[:6] == "color_":
        mask_to_polygon(f)