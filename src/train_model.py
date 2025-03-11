import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
import os

# Correct dataset path format
dataset_path = "C:/Users/Thanm/Desktop/LIVE FACE EMOTION DETECTION/dataset/"

# Load MobileNetV2 Pre-trained Model
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers[-50:]:  # Unfreeze last 50 layers for fine-tuning
    layer.trainable = True


# Count number of emotion classes
num_classes = len(os.listdir(dataset_path))

# Define input layer explicitly
inputs = Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dense(64, activation="relu")(x)
x = Dense(num_classes, activation="softmax")(x)  # Corrected output layer

# Create Model
model = Model(inputs=inputs, outputs=x)

# Compile Model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Image Data Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,      # Rotate images up to 30 degrees
    width_shift_range=0.2,  # Shift images horizontally
    height_shift_range=0.2, # Shift images vertically
    zoom_range=0.2,         # Random zoom
    horizontal_flip=True,   # Flip images randomly
    brightness_range=[0.8, 1.2],  # Vary brightness slightly
    validation_split=0.2    # Keep 20% of data for validation
)


train_generator = datagen.flow_from_directory(dataset_path, target_size=(224, 224),
                                              batch_size=32, class_mode="categorical",
                                              subset="training")
val_generator = datagen.flow_from_directory(dataset_path, target_size=(224, 224),
                                            batch_size=32, class_mode="categorical",
                                            subset="validation")

# Train the Model
model.fit(train_generator, validation_data=val_generator, epochs=30)  # Train for 30 epochs


# Save Model
model.save("models/emotion_model_mobilenet.h5")

print("✅ Model Training Complete! Saved as 'emotion_model_mobilenet.h5'")
