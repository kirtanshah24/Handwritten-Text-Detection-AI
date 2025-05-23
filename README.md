# Handwritten-Text-Detection-AI
This project implements a deep learning model for recognizing handwritten names from grayscale images using a CNN-BiLSTM architecture with CTC loss. It is trained on a publicly available dataset from Kaggle.

## 🧠 Objective
To build a robust model that can accurately recognize handwritten names from grayscale images using deep learning techniques.

## 🏗️ Model Architecture
The model combines Convolutional Neural Networks (CNN) for feature extraction and Bidirectional LSTM (BiLSTM) for sequence modeling. It uses Connectionist Temporal Classification (CTC) for aligning variable-length sequences.

1. CNN Feature Extractor
Conv Block 1: Conv2D(32) → BatchNorm → MaxPool(2x2)

Conv Block 2: Conv2D(64) → BatchNorm → MaxPool(2x2) → Dropout(0.3)

Conv Block 3: Conv2D(128) → BatchNorm → MaxPool(1x2) → Dropout(0.3)

2. Reshape Layer
Converts 3D CNN features into a 2D sequence suitable for RNNs.

3. Bidirectional LSTM
Captures context from both directions to enhance sequence understanding.

4. CTC Loss
Handles alignment of predictions and labels without requiring fixed positioning.

# 🗃️ Dataset
Source: Kaggle - Handwriting Recognition Dataset

Used Files:

written_name_train_v2.csv

written_name_validation_v2.csv

Image folders: train_v2/train/, validation_v2/validation/

# 🧹 Data Preprocessing
Removed samples with missing or "UNREADABLE" labels

Converted labels to uppercase

Resized and rotated images to 256x64

Normalized pixel values to range [0, 1]

# 🔠 Character Encoding
Allowed Characters: A-Z, space, ', -

Max String Length: 24

Labels padded with -1 for CTC compatibility

# ⚙️ Training Configuration
Epochs: 60

Batch Size: 128

Optimizer: Adam

Learning Rate: 0.0001

Loss Function: CTC Loss

# 📊 Evaluation Metrics
Character-Level Accuracy: % of correctly predicted characters

Word-Level Accuracy: % of exact word matches
