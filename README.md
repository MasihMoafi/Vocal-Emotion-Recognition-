# Speech Emotion Recognition

This repository contains a deep learning project for speech emotion recognition using the [speech-emotion-recognition-en dataset](https://www.kaggle.com/datasets/uwrfkaggler/speech-emotion-recognition-en) from Kaggle. The project combines Convolutional Neural Networks (CNN), Bidirectional LSTM, and an Attention mechanism to achieve state-of-the-art performance, with a validation accuracy around 94%.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Demo](#demo)
- [Conclusion](#conclusion)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview
This project demonstrates how to classify speech audio into eight emotion categories:
- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

Data augmentation techniques (noise injection, pitch shifting, time stretching, and time shifting) are applied to improve generalization. The model uses mixup augmentation, early stopping, and learning rate reduction to ensure robust test performance.

## Dataset
The project uses the **RAVDESS** subset from the [speech-emotion-recognition-en dataset](https://www.kaggle.com/datasets/uwrfkaggler/speech-emotion-recognition-en). The dataset contains audio files labeled with emotion codes that are mapped to the eight emotion categories listed above.

## Model Architecture
- **Convolutional Layers:** Extract spatial features from the MFCC representations.
- **Batch Normalization & Dropout:** Regularize the network to prevent overfitting.
- **Bidirectional LSTM Layers:** Capture temporal dependencies in the audio signal.
- **Attention Mechanism:** Weigh temporal features to focus on important segments.
- **Dense Output Layer:** Classifies the input into one of the eight emotion categories using softmax activation.

## Installation
Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/speech-emotion-recognition.git
cd speech-emotion-recognition
pip install -r requirements.txt

