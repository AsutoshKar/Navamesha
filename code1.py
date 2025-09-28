#Firstly importing necessary libraries used for our code.
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Create dataset folders if they do not exist
os.makedirs('microplastics_dataset/train/plastic', exist_ok=True)
os.makedirs('microplastics_dataset/train/non_plastic', exist_ok=True)
os.makedirs('microplastics_dataset/test/plastic', exist_ok=True)
os.makedirs('microplastics_dataset/test/non_plastic', exist_ok=True)

print("Folders created successfully.")

# Check if dataset directories exist
if not os.path.exists('microplastics_dataset/train') or not os.path.exists('microplastics_dataset/test'):
    print("Error: Dataset directories not found. Please check your folder structure.")
    exit()

# Define and compile model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Load your data here (example using ImageDataGenerator)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

training_data = train_datagen.flow_from_directory(
    'microplastics_dataset/train',
    target_size=(128,128),
    batch_size=32,
    class_mode='binary'
)
testing_data = test_datagen.flow_from_directory(
    'microplastics_dataset/test',
    target_size=(128,128),
    batch_size=32,
    class_mode='binary'
)

# Model training
history = model.fit(training_data, validation_data=testing_data, epochs=5)

# TESTING WITH ONE PICTURE
img_path = "/Users/asutoshkar/Documents/python/Navamesha/pic.jpg"
if not os.path.exists(img_path):
    print(f"Error: Test image not found at {img_path}")
    exit()

img = image.load_img(img_path, target_size=(128,128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)[0][0]

if prediction > 0.5:
    print("\nThe image is classified as:Plastic")
else:
    print("\nThe image is classified as:Non-Plastic")

# Plot training accuracy
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()