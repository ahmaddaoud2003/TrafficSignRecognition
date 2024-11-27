import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

# Importing the Test Set and the model
test_dataset = pd.read_csv('data/Test.csv')
cnn = tf.keras.models.load_model('CNN_Traffic_Sign.keras')


labels_test = test_dataset["ClassId"].values
images_test = test_dataset["Path"].values
data_testing = []
for image in images_test:
    image = Image.open('data/' + image)
    image = image.resize([32, 32])
    data_testing.append(np.array(image))
testing_data = np.array(data_testing)


# Predicting results
predictions = np.argmax(cnn.predict(testing_data), axis=-1)

# Displaying the Classification Report
from sklearn.metrics import classification_report, accuracy_score
print(f'Accuracy= {accuracy_score(labels_test, predictions)*100:.2f}%')
print(f'Classification Report: \n{classification_report(labels_test, predictions)}')


# Predicting first 10 signs
plt.figure(figsize=(15,5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(testing_data[i])
    plt.title(f'True: {labels_test[i]}\n Predicted: {predictions[i]}')
    plt.axis('off')
plt.suptitle('First 10 signs', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.show()
