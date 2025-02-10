# Speech Emotion Recognition

This repository contains a deep learning project for speech emotion recognition using the [speech-emotion-recognition-en dataset]([https://www.kaggle.com/datasets/uwrfkaggler/speech-emotion-recognition-en](https://www.kaggle.com/datasets/dmitrybabko/speech-emotion-recognition-en) from Kaggle. The project combines Convolutional Neural Networks (CNN), Bidirectional LSTM, and an Attention mechanism to achieve state-of-the-art performance, with a validation accuracy around 94%.

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

```
git clone https://github.com/yourusername/speech-emotion-recognition.git
cd speech-emotion-recognition
pip install -r requirements.txt
Dependencies include TensorFlow, librosa, numpy, scikit-learn, matplotlib, seaborn, and IPython.
```

Dependencies include TensorFlow, librosa, numpy, scikit-learn, matplotlib, seaborn, and IPython.

Usage
Run the training script to preprocess the data, train the model, and evaluate its performance:

bash
Copy
Edit
python train.py
The script performs:

Data augmentation and MFCC extraction from audio files.
Training with mixup augmentation, early stopping, and learning rate reduction.
Evaluation on a held-out test set.
Visualization of training history and confusion matrix.
Results
The model achieves around 94% validation accuracy, which is excellent for a speech emotion recognition task. Robust test performance is ensured through proper data splits and augmentation techniques.

## Demo
The repository also includes code to demonstrate the model’s performance on a sample audio file. For example, a fearful audio sample is loaded, played, and the predicted emotion is displayed.

When you run the demo section in the notebook or script, it will:

Play a sample audio clip (e.g., a fearful voice).
Print the predicted emotion label (e.g., "fearful").
Conclusion
This project is a strong portfolio piece demonstrating the application of deep learning to speech emotion recognition. With careful data augmentation, a robust model architecture, and effective training strategies, it achieves excellent performance and generalizes well to unseen audio data.

## License
This project is licensed under the MIT License.

## Acknowledgements
Thanks to Kaggle for the [speech-emotion-recognition-en dataset]([https://www.kaggle.com/datasets/uwrfkaggler/speech-emotion-recognition-en](https://www.kaggle.com/datasets/dmitrybabko/speech-emotion-recognition-en) dataset.
Inspiration and research from the deep learning community on CNN-LSTM and attention models.
