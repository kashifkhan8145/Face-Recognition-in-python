import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
# Define dataset path
DATASET_PATH = r"your_path"


# Image parameters
IMG_SIZE = (100, 100)  # Resize all images to 100x100

# Prepare data storage
data = []
labels = []
person_names = []

# Load images
for idx, person_name in enumerate(os.listdir(DATASET_PATH)):
    person_path = os.path.join(DATASET_PATH, person_name)
    person_names.append(person_name)
    
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        
        # Read the image
        img = cv2.imread(img_path)
        img = cv2.resize(img, IMG_SIZE)  # Resize
        img = img / 255.0  # Normalize

        # Store data and labels
        data.append(img)
        labels.append(idx)

# Convert to NumPy arrays
data = np.array(data)
labels = np.array(labels)

# One-hot encode labels
labels = to_categorical(labels, num_classes=len(person_names))

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

print("Dataset loaded successfully!")
print(f"Total samples: {len(data)}")
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(person_names), activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=16)

print("Face recognition model trained successfully!")

# Evaluate the model on test data
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")
import matplotlib.pyplot as plt

# Select a test image
idx = np.random.randint(0, len(X_test))
test_img = X_test[idx]

# Predict class
prediction = model.predict(np.expand_dims(test_img, axis=0))
predicted_label = np.argmax(prediction)

# Display result
plt.imshow(test_img)
plt.title(f"Predicted: {person_names[predicted_label]}")
plt.show()

