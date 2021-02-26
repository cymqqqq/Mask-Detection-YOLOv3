import cv2
import numpy as np
import os
import argparse
import time

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights_file", required=True, help="path to YOLO weights file")
ap.add_argument("-cfg", "--config_file", required=True, help="path to YOLO config file")
ap.add_argument("-n", "--obj.names_file", required=True, help="path to YOLO obj.names file")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.5, help="threshold when applying non-maximum suppression")
ap.add_argument("-iou", "--intersection_union", type=float, default=0.5, help="threshold for intersection over union")
ap.add_argument("-o", "--save_frame_output", required=False, help="path of output images")
args = vars(ap.parse_args())

# py video_detector.py -w C:\Users\Acer\PycharmProjects\mask_detection_YOLOv3\mask_detection_v1\yolov3_last.weights -cfg C:\Users\Acer\PycharmProjects\mask_detection_YOLOv3\mask_detection_v1\darknet\cfg\yolov3.cfg -n C:\Users\Acer\PycharmProjects\mask_detection_YOLOv3\mask_detection_v1\darknet\data\obj.names -o C:\Users\Acer\PycharmProjects\mask_detection_YOLOv3\frame
# py video_detector.py -w C:\Users\Acer\PycharmProjects\mask_detection_YOLOv3\mask_detection_v2\yolov3_last.weights -cfg C:\Users\Acer\PycharmProjects\mask_detection_YOLOv3\mask_detection_v2\darknet\cfg\yolov3.cfg -n C:\Users\Acer\PycharmProjects\mask_detection_YOLOv3\mask_detection_v2\darknet\data\obj.names -o C:\Users\Acer\PycharmProjects\mask_detection_YOLOv3\frame

net = cv2.dnn.readNet(args['weights_file'], args['config_file'])
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

classes = open(args['obj.names_file']).read().strip().split("\n")

CONFIDENCE = args['confidence']
SCORE_THRESHOLD = args['threshold']
IOU_THRESHOLD = args['intersection_union']

IMAGE_PATH = args['save_frame_output']
if IMAGE_PATH != '' and IMAGE_PATH is not None:
    os.mkdir(IMAGE_PATH)

# Format of colours in BGR. Red colour when no mask, green colour when there is mask
colors = [[0, 255, 0], [0, 0, 255]]

cap = cv2.VideoCapture(0)

count = 1

while True:
    ret, img = cap.read()

    # resize the frame
    img = cv2.resize(img, (1000, 700))

    # img = cv2.resize(img, (img.shape[1]*3, img.shape[0]*3))
    h, w = img.shape[:2]

    # create 4D blob
    # normalize, scale and reshape this image to be suitable as an input to the neural network
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # print("img.shape:", img.shape)
    # print("blob.shape:", blob.shape)

    # sets the blob as the input of the network
    net.setInput(blob)

    # get all the layer names
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # measure how long it took in seconds
    start = time.perf_counter()
    layer_outputs = net.forward(ln)
    time_took = time.perf_counter() - start
    print(f"Time took: {time_took:.2f}s")

    font_scale = 1
    thickness = 1
    boxes, confidences, class_ids = [], [], []

    # loop over each of the layer outputs
    for output in layer_outputs:
        # loop over each of the object detections
        for detection in output:
            # extract the class id (label) and confidence (as a probability) of the current object detection
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > CONFIDENCE:
                # scale the bounding box coordinates back relative to the size of the image
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # perform the non maximum suppression given the scores defined before
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)

    border_size = 50
    border_text_color = [255, 255, 255]
    # Add top-border to image to display stats
    img = cv2.copyMakeBorder(img, border_size, 0, 0, 0, cv2.BORDER_CONSTANT)

    # calculate count values
    filtered_classids = np.take(class_ids, idxs)

    mask_count = (filtered_classids == 0).sum()
    nomask_count = (filtered_classids == 1).sum()

    # display count
    text = "Mask: {}  No Mask: {}".format(mask_count, nomask_count)
    cv2.putText(img, text, (0, int(border_size - 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, border_text_color, 2)

    # ensure at least one detection exists
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y = boxes[i][0], boxes[i][1] + border_size
            w, h = boxes[i][2], boxes[i][3]

            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in colors[class_ids[i]]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color=color, thickness=thickness*2)

            text = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            print(text)
            cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color=color, thickness=thickness)

            # calculate text width & height to draw the transparent boxes as background of the text
            (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
            text_offset_x = x
            text_offset_y = y
            box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))

            overlay = img.copy()
            cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)

            # now put the text (label: confidence %)
            cv2.putText(img, text, (w - 500, int(border_size - 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        border_text_color, 1)

    if IMAGE_PATH != '' and IMAGE_PATH is not None:
        cv2.imwrite(os.path.join(IMAGE_PATH, str(count).zfill(2) + ".jpg"), img)

    cv2.imshow('Detections', img)

    count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


