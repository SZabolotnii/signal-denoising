# Logarithmic Spectrogram Preprocessing Pipeline

This directory contains code for preprocessing signals into logarithmic spectrograms before model training.

## Overview

This approach pre-generates a dataset of logarithmic spectrogram images rather than performing signal-to-spectrogram
conversion during training. This design offers several advantages:

- **Improved Training Performance**: Eliminates computational overhead from real-time transformations during training
- **Better Visualization**: Enables easier inspection and analysis of the data throughout the training and evaluation
  process
- **Data Augmentation Support**: Facilitates augmentation techniques to expand the training dataset

## Workflow

1. **Dataset Generation**: Convert raw audio signals to logarithmic spectrogram images
2. **Augmentation**: Apply data augmentation techniques to increase dataset size and diversity
3. **Training**: Use the preprocessed spectrograms with the U-Net model architecture

## Model Architecture

The same U-Net model architecture is utilized for training on these preprocessed spectrograms.