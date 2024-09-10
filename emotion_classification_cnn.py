

#run dev for oversampling
import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input, UpSampling2D, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from sklearn.utils import resample
from collections import Counter

# Constants
picture_size = 48
batch_size = 128
no_of_classes = 7
folder_path = '/content/Emotion_Detection_CNN-main/FER/images/'

# Data augmentation
datagen_train = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.35,
                                   zoom_range=0.35,
                                   horizontal_flip=True,
                                   rotation_range=35,
                                   width_shift_range=0.35,
                                   height_shift_range=0.35,
                                   brightness_range=[0.8, 1.2])

datagen_val = ImageDataGenerator(rescale=1./255)

# Custom Balanced Data Generator
class BalancedDataGenerator(Sequence):
    def __init__(self, directory, classes, batch_size, target_size, datagen, shuffle=True):
        self.directory = directory
        self.classes = classes
        self.batch_size = batch_size
        self.target_size = target_size
        self.datagen = datagen
        self.shuffle = shuffle

        # Load the image paths and labels
        self.image_paths, self.labels = self.load_data(self.directory, self.classes)

        # Balance the dataset by oversampling smaller classes
        self.image_paths, self.labels = self.balance_classes(self.image_paths, self.labels)

        self.indexes = np.arange(len(self.image_paths))
        self.on_epoch_end()

    def load_data(self, directory, classes):
        image_paths = []
        labels = []

        for class_index, class_name in enumerate(classes):
            class_dir = f"{directory}/{class_name}"
            class_image_paths = [f"{class_dir}/{img}" for img in os.listdir(class_dir)]
            image_paths.extend(class_image_paths)
            labels.extend([class_index] * len(class_image_paths))

        return np.array(image_paths), np.array(labels)

    def balance_classes(self, image_paths, labels):
        class_counts = Counter(labels)
        max_class_count = max(class_counts.values())

        balanced_image_paths = []
        balanced_labels = []

        for class_index in np.unique(labels):
            class_image_paths = image_paths[labels == class_index]
            class_labels = labels[labels == class_index]

            # Oversample the class if it has fewer examples
            if len(class_image_paths) < max_class_count:
                class_image_paths = resample(class_image_paths, replace=True, n_samples=max_class_count, random_state=42)
                class_labels = [class_index] * max_class_count

            balanced_image_paths.extend(class_image_paths)
            balanced_labels.extend(class_labels)

        return np.array(balanced_image_paths), np.array(balanced_labels)

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of image paths and labels for this batch
        batch_image_paths = self.image_paths[batch_indexes]
        batch_labels = self.labels[batch_indexes]

        # Generate data
        X, y = self.__data_generation(batch_image_paths, batch_labels)

        return X, y

    def on_epoch_end(self):
        # Updates indexes after each epoch
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_image_paths, batch_labels):
        # Initialization
        X = np.empty((self.batch_size, *self.target_size, 1))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, (img_path, label) in enumerate(zip(batch_image_paths, batch_labels)):
            # Load grayscale image
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, self.target_size)

            # Expand dimensions to add channel (for grayscale images, channel=1)
            img = np.expand_dims(img, axis=-1)

            # Apply data augmentation (random transformations)
            img = self.datagen.random_transform(img)

            # Store sample and corresponding label
            X[i,] = img
            y[i] = label

        # One-hot encoding for labels
        y = tf.keras.utils.to_categorical(y, num_classes=len(self.classes))

        return X, y

# Define the class names
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Initialize the balanced data generator
train_generator = BalancedDataGenerator(folder_path + "train", classes, batch_size=batch_size, target_size=(picture_size, picture_size), datagen=datagen_train)
val_generator = BalancedDataGenerator(folder_path + "validation", classes, batch_size=batch_size, target_size=(picture_size, picture_size), datagen=datagen_val, shuffle=False)

# Feature Pyramid and Expression Disentanglement Model
def feature_pyramid_disentangle(input_tensor):
    # Block 1
    conv1 = Conv2D(64, (3, 3), padding='same', activation='relu')(input_tensor)
    conv1 = BatchNormalization()(conv1)
    conv1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv1 = Dropout(0.30)(conv1)  # Adjusted dropout

    # Block 2
    conv2 = Conv2D(128, (5, 5), padding='same', activation='relu')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv2 = Dropout(0.30)(conv2)  # Adjusted dropout

    # Block 3
    conv3 = Conv2D(256, (3, 3), padding='same', activation='relu')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv3 = Dropout(0.30)(conv3)  # Adjusted dropout

    # Block 4
    conv4 = Conv2D(512, (3, 3), padding='same', activation='relu')(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv4 = Dropout(0.30)(conv4)  # Adjusted dropout

    # Upsampling and Concatenation
    up1 = UpSampling2D(size=(conv3.shape[1] // conv4.shape[1], conv3.shape[2] // conv4.shape[2]))(conv4)
    up1 = Concatenate()([up1, conv3])

    up2 = UpSampling2D(size=(conv2.shape[1] // up1.shape[1], conv2.shape[2] // up1.shape[2]))(up1)
    up2 = Concatenate()([up2, conv2])

    up3 = UpSampling2D(size=(conv1.shape[1] // up2.shape[1], conv1.shape[2] // up2.shape[2]))(up2)
    up3 = Concatenate()([up3, conv1])

    return up3

# Main Model
input_layer = Input(shape=(48, 48, 1))
x = feature_pyramid_disentangle(input_layer)

# Flatten and Dense layers
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.30)(x)  # Adjusted dropout

x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.30)(x)  # Adjusted dropout

output_layer = Dense(no_of_classes, activation='softmax')(x)

# Model definition
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
opt = Adam(learning_rate=0.0001, weight_decay=0.00001)  # Adjusted learning rate
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Checkpoints and Early Stopping
checkpoint = ModelCheckpoint('/content/Emotion_Detection_CNN-main/models/best_model.keras',
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min')

early_stopping = EarlyStopping(monitor='val_loss',
                               patience=5,
                               verbose=1,
                               restore_best_weights=True)

# Train the model using the balanced generator
history = model.fit(train_generator,
                    validation_data=val_generator,
                    epochs=30,
                    callbacks=[checkpoint, early_stopping])

# Save the final trained model
model.save('/content/Emotion_Detection_CNN-main/models/saved_model.keras')

# Plot accuracy and loss graphs
plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

