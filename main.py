import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Activation, BatchNormalization, Rescaling
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import image_dataset_from_directory
# label = 1 (penguin), label = 2 (turtle)
CLASS_LABELS = {1: 'penguin', 2: 'turtle'}
CLASS_NAME = ['penguin', 'turtle']
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
BATCH_SIZE = 10
IMAGE_CHANNELS = 3

path = os.getcwd() # Path should = something/COMP9517-project/
train_image_dir = os.path.join(path, 'sorted_data','train')
valid_image_dir = os.path.join(path, 'sorted_data','valid')
test_image_dir = os.path.join(path, 'sorted_data','test')


train_dataset = image_dataset_from_directory(
    train_image_dir,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

validation_dataset = image_dataset_from_directory(
    valid_image_dir,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

test_dataset = image_dataset_from_directory(
    test_image_dir,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# plt.figure(figsize=(10,10))
# for images, labels in train_dataset.take(1):
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     plt.title(CLASS_NAME[labels[i]])
#     plt.axis("off")
# plt.show()

# build the model
model = Sequential()
model.add(Rescaling(1./255))
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2)) # prevent overfitting

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2)) # prevent overfitting

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2)) # prevent overfitting

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2)) # prevent overfitting
model.add(Dense(2, activation='softmax')) # 2 because we have 2 classes

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.summary()

# fit the model with training data
history = model.fit(
    train_dataset,
    epochs=100,
    validation_data=validation_dataset,
)
# Evaluate the model and visualize the results
test_loss, test_accuracy = model.evaluate(validation_dataset)
print(f'Valid Loss: {test_loss}')
print(f'Valid Accuracy: {test_accuracy}')

# test_loss, test_accuracy = model.evaluate(test_dataset)
# print(f'Test Loss: {test_loss}')
# print(f'Test Accuracy: {test_accuracy}')

plt.figure(figsize=(10, 10))
for images, labels in test_dataset.take(1):
    classifications = model(images)
    print(classifications)
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype('uint8'))
        index = np.argmax(classifications[i])
        plt.title("Pred: " + CLASS_NAME[index] + " Actual: " + CLASS_NAME[labels[i]])

plt.show()