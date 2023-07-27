# COMP9517-project 
classification and detection of turtles and penguins

## Description
- This project aims to classify and detect marine life which are turtles and penguins, using computer vision techniques. It utilizes deep learning models of yolov5 and fastrcnn for both classification and object detection tasks, enabling the automatic recognition and localization of turtles and penguins in images. In order to improve the accuracy of training, this project still uses some image preprocessing.

## Features
- Classification: The project includes pre-trained models capable of accurately classifying images into either "turtle" or "penguin" categories.
- Object Detection: Utilizing object detection algorithms, the system identifies and draws bounding boxes around the detected turtles and penguins in real-time.


## Usage

1. Install the required dependencies listed in `requirements.txt` using the following command:

pip install -r requirements.txt

2. Do the image preprocess.

run the file image_preprocess.ipynb in folder demo to preprocess the dataset

3. Do the image Augmentation

run the file image_Augmentation.ipynb in folder demo to augmentation the dataset

4. when after the trainning delete all preprocess image

run the file delete_process_image.ipynb in folder demo to delete the dataset preprocess



