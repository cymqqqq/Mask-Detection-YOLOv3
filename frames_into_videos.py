import cv2
import glob

img_array = []
for filename in glob.glob(r'C:\Users\Acer\PycharmProjects\mask_detection_YOLOv3\mask_detection_v1\frame\*.jpg'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter(r'C:\Users\Acer\PycharmProjects\mask_detection_YOLOv3\output_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 2, size)
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
