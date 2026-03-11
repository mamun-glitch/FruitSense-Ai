"""
FruitSense AI: Smart Fruit Freshness Detection System
-----------------------------------------------------
Dataset: Fruits Fresh and Rotten for Classification (Kaggle)
Link: https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification

Description:
- The base dataset used in this project was the Fruits Fresh and Rotten dataset from Kaggle,
  which originally includes apple, banana, and orange images classified as fresh or rotten.
- To improve the system’s capability and make the model more versatile, additional training
  images were incorporated into the dataset.
- Specifically, watermelon, jackfruit, and strawberry were added to create a more robust and
  comprehensive classification system.
- This expansion allows the model to generalize better across a wider variety of fruit textures,
  shapes, and colors commonly found in markets.
- The final dataset now supports six distinct fruit categories, providing a more realistic and
  practical tool for automated freshness detection.

This script trains a Convolutional Neural Network (CNN) that classifies fruits
(Apples, Bananas, Oranges, Watermelon, Jackfruit, Strawberry) as either Fresh or Rotten.
It is designed to handle the specific classes found in the dataset and map them to a binary
Fresh/Rotten outcome for user-friendly prediction.
"""

import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix

# --- Configuration & Hyperparameters ---
IMG_WIDTH, IMG_HEIGHT = 128, 128
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.001

# Dataset paths
TRAIN_DIR = 'dataset/train'
# Note:
# The 'test' directory is used as the validation set during training.
# In a more advanced version, this should be split into:
# train / validation / test
TEST_DIR = 'dataset/test'

# Output files
MODEL_PATH = 'fruit_freshness_model.h5'
CLASS_INDICES_PATH = 'class_indices.json'
TRAINING_GRAPH_PATH = 'training_history.png'
CONFUSION_MATRIX_PATH = 'confusion_matrix.png'
CLASSIFICATION_REPORT_PATH = 'classification_report.txt'


def create_data_generators():
    """
    Creates training and test data generators.
    Data augmentation is applied only to the training set to reduce overfitting.
    """
    print("Initializing data generators...")

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, test_generator


def build_network(num_classes):
    """
    Builds the CNN architecture.
    """
    print(f"Building model with {num_classes} output classes...")

    model = Sequential([
        Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)),

        # Block 1
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.2),

        # Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Block 3
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        # Dense layers
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),

        # Output layer
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def plot_history(history):
    """
    Plots and saves training/validation accuracy and loss graphs.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(15, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training vs Validation Accuracy')
    plt.grid(True)

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training vs Validation Loss')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(TRAINING_GRAPH_PATH)
    print(f"Saved training history graph to '{TRAINING_GRAPH_PATH}'")
    plt.show()
    plt.close()


def save_classification_report(report_text):
    """
    Saves the classification report to a text file.
    """
    with open(CLASSIFICATION_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("Fruit Freshness Classification Report\n")
        f.write("=====================================\n\n")
        f.write(report_text)
        f.write("\n\nLIMITATION NOTE:\n")
        f.write(
            "Due to time and dataset structure constraints, the provided test set was used "
            "as the validation set during training and for final evaluation. In future work, "
            "a separate hold-out test set would be created for stricter performance assessment.\n"
        )
    print(f"Saved classification report to '{CLASSIFICATION_REPORT_PATH}'")


def save_confusion_matrix(true_classes, pred_classes, class_labels):
    """
    Generates and saves the confusion matrix.
    """
    cm = confusion_matrix(true_classes, pred_classes)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_labels,
        yticklabels=class_labels
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    print(f"Saved confusion matrix to '{CONFUSION_MATRIX_PATH}'")
    plt.show()
    plt.close()


def train_model():
    """
    Trains the CNN model and evaluates it.
    """
    if not os.path.exists(TRAIN_DIR):
        print(f"Dataset not found at '{TRAIN_DIR}'. Please download and extract the dataset correctly.")
        return

    train_gen, test_gen = create_data_generators()

    class_indices = train_gen.class_indices
    print("Class mapping:", class_indices)

    with open(CLASS_INDICES_PATH, 'w', encoding="utf-8") as f:
        json.dump(class_indices, f, indent=4)
    print(f"Saved class mapping to '{CLASS_INDICES_PATH}'")

    model = build_network(num_classes=len(class_indices))
    model.summary()

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    ]

    print("Starting training...")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=test_gen,
        callbacks=callbacks
    )

    plot_history(history)

    print("Evaluating model...")
    loss, accuracy = model.evaluate(test_gen)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    print("Generating predictions for classification report and confusion matrix...")
    predictions = model.predict(test_gen)
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = test_gen.classes
    class_labels = list(class_indices.keys())

    report = classification_report(true_classes, pred_classes, target_names=class_labels)
    print("\nClassification Report:")
    print(report)

    save_classification_report(report)
    save_confusion_matrix(true_classes, pred_classes, class_labels)

    model.save(MODEL_PATH)
    print(f"Model saved successfully to '{MODEL_PATH}'")


def format_prediction_label(predicted_class_name):
    """
    Converts raw class names like 'freshapples' into display labels like 'Fresh Apple'.
    """
    raw_name = predicted_class_name.lower().replace("fresh", "").replace("rotten", "").strip()

    fruit_name_map = {
        "apples": "Apple",
        "bananas": "Banana",
        "oranges": "Orange",
        "watermelon": "Watermelon",
        "jackfruit": "Jackfruit",
        "strawberry": "Strawberry"
    }

    display_name = fruit_name_map.get(raw_name, raw_name.title())

    if "fresh" in predicted_class_name.lower():
        freshness_state = "Fresh"
    elif "rotten" in predicted_class_name.lower():
        freshness_state = "Rotten"
    else:
        freshness_state = "Unknown"

    return f"{freshness_state} {display_name}"


def predict_image(image_path):
    """
    Predicts the class of a single image.
    """
    if not os.path.exists(MODEL_PATH):
        print(f"Model file '{MODEL_PATH}' not found. Please train the model first.")
        return

    if not os.path.exists(image_path):
        print(f"Image file '{image_path}' not found.")
        return

    if os.path.exists(CLASS_INDICES_PATH):
        with open(CLASS_INDICES_PATH, 'r', encoding="utf-8") as f:
            class_indices = json.load(f)
        idx_to_class = {v: k for k, v in class_indices.items()}
    else:
        print(f"Warning: '{CLASS_INDICES_PATH}' not found. Using fallback labels.")
        fallback_labels = [
            'freshapples', 'freshbananas', 'freshjackfruit', 'freshoranges',
            'freshstrawberry', 'freshwatermelon', 'rottenapples', 'rottenbananas',
            'rottenjackfruit', 'rottenoranges', 'rottenstrawberry', 'rottenwatermelon'
        ]
        idx_to_class = {i: label for i, label in enumerate(fallback_labels)}

    print(f"Loading model from '{MODEL_PATH}'...")
    model = load_model(MODEL_PATH)

    img = load_img(image_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    print("Predicting...")
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction[0])
    confidence = prediction[0][class_idx]

    predicted_class_name = idx_to_class.get(class_idx, "Unknown")
    label = format_prediction_label(predicted_class_name)

    print(f"\nResult: {label}")
    print(f"Confidence: {confidence * 100:.2f}%")

    plt.imshow(img)
    plt.title(f"{label}\nConfidence: {confidence * 100:.2f}%")
    plt.axis('off')
    plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="FruitSense AI: Fruit Freshness Detection System")
    parser.add_argument('--train', action='store_true', help="Train the CNN model")
    parser.add_argument('--predict', type=str, help="Path to an image file to classify")

    args = parser.parse_args()

    if args.train:
        train_model()
    elif args.predict:
        predict_image(args.predict)
    else:
        print("Usage:")
        print("  Train model: python fruit_freshness_cnn.py --train")
        print("  Predict:     python fruit_freshness_cnn.py --predict path/to/image.jpg")


if __name__ == "__main__":
    main()
