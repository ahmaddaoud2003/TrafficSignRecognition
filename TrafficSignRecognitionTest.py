import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

# Importing the Test Set and the model
test_dataset = pd.read_csv('data/Test.csv')
cnn = tf.keras.models.load_model('CNN_Traffic_Sign.keras')


# Dictionary for labels --> not necessary only for output purposes
labels = {0: 'Speed Limit 20km/h', 1: 'Speed Limit 30km/h', 2: 'Speed Limit 50km/h', 3: 'Speed Limit 60km/h',
          4: 'Speed Limit 70km/h', 5: 'Speed Limit 80km/h', 6: 'End of 80km/h speed limit', 7: 'Speed Limit 100km/h',
          8: 'Speed Limit 120km/h', 9: 'No Overtaking', 10: 'No Overtaking of Trucks', 11: 'Priority',
          12: 'Priority Road', 13: 'Yield', 14: 'Stop', 15: 'Road Closed', 16: 'Trucks Prohibited', 17: 'Do Not Enter',
          18: 'General Danger', 19: 'Left Curve', 20: 'Right Curve', 21: 'Double Curve', 22: 'Uneven Road',
          23: 'Slippery Road', 24: 'Road Narrows', 25: 'Construction Work', 26: 'Traffic Signal Ahead',
          27: 'Pedestrian Crossing', 28: 'Watch for Children', 29: 'Bicycle Crossing', 30: 'Ice/Snow',
          31: 'Wild Animal Crossing', 32: 'End of All Restrictions', 33: 'Turn Right Ahead', 34: 'Turn Left Ahead',
          35: 'Ahead Only', 36: 'Ahead or Right Turn Only', 37: 'Ahead or Left Turn Only', 38: 'Pass By on Right',
          39: 'Pass By on Left', 40: 'Roundabout', 41: 'End of No Overtaking Zone',
          42: 'End of No Overtaking for Trucks'}


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


# Predicting 10 random signs
random_signs_index = random.sample(range(len(testing_data)), k=10)
plt.figure(figsize=(30,10))
for index, i in enumerate(random_signs_index):
    image = testing_data[i]
    true_label = labels[labels_test[i]]
    predicted_label = labels[np.argmax(cnn.predict(np.expand_dims(image, axis=0)))]
    plt.subplot(2, 5, index+1)
    plt.imshow(image)
    plt.title(f'True Label: {true_label}\nPredicted Labels: {predicted_label}', fontsize=18, fontweight='bold')
    plt.axis('off')
plt.suptitle('10 Random signs', fontsize=40, fontweight='bold')
plt.tight_layout()
plt.show()

