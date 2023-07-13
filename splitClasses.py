import os
import json
import cv2
import shutil

# label = 1 (penguin), label = 2 (turtle)
CLASS_LABELS = {1: 'penguin', 2: 'turtle'}

path = os.getcwd() # Path should = something/COMP9517-project/
train_image_dir = os.path.join(path, 'train')
train_image_path = os.path.join(path, 'train', 'train')
train_images = [os.path.join(train_image_path, f) for f in os.listdir(train_image_path)]
valid_image_dir = os.path.join(path, 'valid')
valid_image_path = os.path.join(path, 'valid', 'valid')
valid_images = [os.path.join(valid_image_path, f) for f in os.listdir(valid_image_path)]

new_train_image_path = os.path.join(path, 'sorted_data', 'train')
new_valid_image_path = os.path.join(path, 'sorted_data', 'valid')

# get train and valid images
train_annotations_path = os.path.join(path, 'train_annotations')
train_annotation_file = open(train_annotations_path, 'r')
train_annotation = train_annotation_file.read()
train_annotation = json.loads(train_annotation)
train_annotation_file.close()

validation_annotations_path = os.path.join(path, 'valid_annotations')
validation_annotation_file = open(validation_annotations_path, 'r')
validation_annotation = validation_annotation_file.read()
validation_annotation = json.loads(validation_annotation)
validation_annotation_file.close()

#get label for each image
def getLabels(annotations):
    labels = []
    for annotation in annotations:
        label = annotation['category_id']
        label = CLASS_LABELS[label]
        labels.append(label)
    return labels

train_labels = getLabels(train_annotation)
valid_labels = getLabels(validation_annotation)

# for i in range(3):
#     text = f"image {i} label = {train_labels[i]}"
#     cv2.imshow(text, cv2.imread(train_images[i]))
#     cv2.waitKey(0)
    
train_image_label_pair = list(zip(train_images, train_labels))
valid_image_label_pair = list(zip(valid_images, valid_labels))

# print(image_label_pair[:4])

def rename_and_move_files(image_label_pair, source_path, dest_path):
    class_dirs = ['penguin', 'turtle']
    # check and create class directories if not exist
    for class_dir in class_dirs:
        class_path = os.path.join(dest_path, class_dir) 
        if not os.path.exists(class_path):
            os.makedirs(class_path)
    
    #iterate image-label pair and rename + move to new directory
    for image_name, label in image_label_pair:
        source_path = os.path.join(source_path, image_name)
        
        # print(f'source_path = {source_path}')
        
        if label == 'penguin':
            class_dir = 'penguin'
        elif label == 'turtle':
            class_dir = 'turtle'
        else:
            continue
        
        image_id = image_name.split('\\')[-1]
        
        new_name = f'{class_dir}_{image_id}'
        
        # Create the destination path
        destination_path = os.path.join(dest_path, class_dir, new_name)
        
        # print(f'destination_path = {destination_path}')
        
        # do nothing if file already exists
        if os.path.exists(destination_path):
            continue
        
        shutil.copy(source_path, destination_path)

rename_and_move_files(train_image_label_pair, train_image_path, new_train_image_path)
rename_and_move_files(valid_image_label_pair, valid_image_path, new_valid_image_path)