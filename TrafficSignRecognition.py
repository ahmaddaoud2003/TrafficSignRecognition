import pandas as pd
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split

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
train_images, val_images, train_labels, val_labels = train_test_split(data, labels, test_size=0.15, random_state=42)

# Normalize pixel values to [0, 1]
train_images = train_images.astype('float32') / 255.0
val_images = val_images.astype('float32') / 255.0

# Categorizing labels (One Hot Encoding)
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=43)
val_labels = tf.keras.utils.to_categorical(val_labels, num_classes=43)

# Data Augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,           # Random rotation up to 10 degrees
    width_shift_range=0.1,       # Random horizontal shift
    height_shift_range=0.1,      # Random vertical shift
    shear_range=0.1,            # Random shear transformation
    zoom_range=0.1,             # Random zoom
    brightness_range=[0.8, 1.2], # Random brightness adjustment
    fill_mode='nearest',         # Fill mode for transformations
    horizontal_flip=False        # No horizontal flip (traffic signs should maintain orientation)
)

# Fit the data generator on training data
datagen.fit(train_images)

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

# Training with data augmentation
history = cnn.fit(
    datagen.flow(train_images, train_labels, batch_size=100),
    steps_per_epoch=len(train_images) // 100,
    validation_data=(val_images, val_labels),
    epochs=15,
    verbose=1
)

# Saving the model
cnn.save('CNN_Traffic_Sign_Augmented.keras')

# Optional: Plot training history
import matplotlib.pyplot as plt

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot training & validation accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot training & validation loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

# Uncomment to plot training history
# plot_training_history(history)
