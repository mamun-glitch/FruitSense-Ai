# FruitSense AI: Smart Fruit Freshness Detection System

## Overview
FruitSense AI is a computer vision and deep learning project developed to classify fruit images as **Fresh** or **Rotten**. The system uses a Convolutional Neural Network (CNN) to analyze visual features such as color consistency, surface texture, and signs of decay.

This project demonstrates how artificial intelligence can be applied to food quality monitoring and agricultural inspection.

---

## Dataset
**Dataset Name:** Fruits Fresh and Rotten for Classification  
**Source:** Kaggle  
**Link:** https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification

### Original Dataset Classes
- Apple
- Banana
- Orange

### Additional Fruit Classes Added in This Project
- Watermelon
- Jackfruit
- Strawberry

The final system supports six fruit categories, each with both **fresh** and **rotten** samples.

---

## Features
- Image preprocessing and normalization
- Data augmentation for improved generalization
- CNN-based multi-class classification
- Binary freshness interpretation for user-friendly output
- Training and validation accuracy graph
- Confusion matrix generation
- Classification report generation
- Single-image prediction support

---

## Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## Project Structure
```text
FruitSense-AI/
├── fruit_freshness_cnn.py
├── requirements.txt
├── README.md
├── .gitignore
