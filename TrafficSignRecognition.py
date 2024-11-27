import pandas as pd
import os
import numpy as np
import tensorflow as tf
from PIL import Image


# Importing datasets
training_dataset = pd.read_csv("data/Train.csv")
test_dataset = pd.read_csv("data/Test.csv")


# Converting to Numpy array
data = []
labels = []
classes = 43
cur_path = os.getcwd()
for i in range(classes):
    path = os.path.join('data/Train', str(i))
    images = os.listdir(path)

    for image in images:
        try:
            image = Image.open(path + '/' + image)
            image = image.resize((32, 32))
            image = np.array(image)
            data.append(image)
            labels.append(i)
        except:
            print("Image Error")

data = np.array(data)
labels = np.array(labels)


# Splitting the training set into a training set and a validation set
from sklearn.model_selection import train_test_split
train_images, val_images, train_labels, val_labels = train_test_split(data, labels, test_size=0.15, random_state=42)

# Categorizing labels (One Hot Encoding)
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=43)
val_labels = tf.keras.utils.to_categorical(val_labels, num_classes=43)


# Creating the convolutional neural network

# Initializing the CNN
cnn = tf.keras.Sequential()

# Adding the Convolutional and Pooling Layers --> 3 convolutional layers & 2 pooling layers
cnn.add(tf.keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=(32, 32, 3)))
cnn.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu'))
cnn.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Dropout(0.15))
cnn.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
cnn.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu'))
cnn.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Dropout(0.2))

# Flattening to input in Neural Network
cnn.add(tf.keras.layers.Flatten())

# Connecting to Neural Network
cnn.add(tf.keras.layers.Dense(512, activation='relu'))  # First hidden layer
cnn.add(tf.keras.layers.Dropout(0.25))
cnn.add(tf.keras.layers.Dense(43, activation='softmax'))  # Output Layer

# Compiling the Neural Network
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn.fit(x=train_images, y=train_labels, validation_data=(val_images, val_labels), epochs=15, batch_size=100)

# Saving the model
cnn.save('CNN_Traffic_Sign.keras')
