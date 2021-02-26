import os
import random

#####################################
# COUNT NUMBER OF IMAGES AND LABELS #
#####################################
mask_count = 0
no_mask_count = 0
total_images = 0
PATH = r'C:\Users\Acer\PycharmProjects\mask_detection_YOLOv3\mask_detection_less_dataset\darknet\data\mask_images'
for file in os.listdir(PATH):
    if file != 'classes.txt' and file.split('.')[1] == 'txt':
        with open(os.path.join(PATH, file), 'r') as r:
            for line in r.readlines():
                # class 0 represents mask, while class 1 represents no mask
                if line[0] == '0':
                    mask_count += 1
                elif line[0] == '1':
                    no_mask_count += 1
        total_images += 1
print(mask_count)
print(no_mask_count)
print(total_images)

#############################################################
# CHECK THAT EACH IMAGE HAS A TXT FILE, ELSE RAISE AN ERROR #
#############################################################
img_list = []
for image in os.listdir(PATH):
    # Ensure images are in correct format
    if image.split('.')[-1] in ['jpg', 'jpeg', 'png']:
        img_list.append(image.split('.')[0])

for txt_file in os.listdir(PATH):
    if txt_file.split('.')[-1] == 'txt':
        if txt_file.split('.')[0] in img_list:
            img_list.remove(txt_file.replace('.txt', ''))
print('ERROR IMAGES', img_list)

###########################################
# SPLIT INTO TRAINING AND TESTING DATASET #
###########################################
list = []
num_images = 0
for image in os.listdir(PATH):
    if image.split('.')[-1] != 'txt':
        list.append(image)
        num_images += 1
random.shuffle(list)

# 80% of images will be for training, 20% will be for testing
train_data = list[:int(round(num_images*0.8))]
test_data = list[int(round(num_images*0.8)):]

with open(r'C:\Users\Acer\PycharmProjects\mask_detection_YOLOv3\mask_detection_less_dataset\darknet\data\train.txt', 'w') as train:
    for img in train_data:
        train.write('data/mask_images/' + img + '\n')

with open(r'C:\Users\Acer\PycharmProjects\mask_detection_YOLOv3\mask_detection_less_dataset\darknet\data\test.txt', 'w') as train:
    for img in test_data:
        train.write('data/mask_images/' + img + '\n')
