# University Computer Vision Labs

This repository contains three lab assignments from a university course, focusing on image classification and segmentation using PyTorch.

## Lab Overview

| Lab | Title                                      | Focus                                         |
|-----|--------------------------------------------|-----------------------------------------------|
| 1   | Pretrained ResNet50 Image Classification   | Classification with threshold tuning          |
| 2   | Custom PyTorch Classifier                  | CNN training and evaluation                   |
| 3   | CamVid Semantic Segmentation               | Pixel-wise image segmentation                 |

---

## Lab1

### Lab 1 – Image Classification with Pretrained Model (ResNet50)

This project uses a pretrained ResNet50 model to classify images from the OpenImages dataset into three classes: Bread, Cat, and Banana.

Goals:
- Use a pretrained model (ResNet50)
- Classify 1000 images
- Evaluate model performance using:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
- Adjust classification thresholds and observe impact on results

Dataset:
- 1000 images from OpenImages

Tools:
- PyTorch
- torchvision
- sklearn

---

## Lab2

### Lab 2 – Custom Image Classifier with PyTorch

This lab focuses on building and training a custom image classifier to distinguish between Cats, Cars, and Dogs.

Goals:
- Load and preprocess image data
- Train a CNN model using PyTorch
- Split data into training and testing sets
- Evaluate model performance with:
  - Confusion matrix
  - Accuracy
  - Precision
  - Recall
  - F1 Score

Dataset:
- Images from OpenImages

Tools:
- PyTorch
- torchvision
- sklearn
- matplotlib

---

## Lab3

### Lab 3 – Semantic Segmentation using CamVid Dataset

This lab prepares a semantic segmentation task using the CamVid dataset for classifying image pixels into categories like road, car, pedestrian, etc.

Goals:
- Load and extract CamVid dataset
- Prepare data for segmentation
- Train a segmentation model (e.g. U-Net)
- Evaluate performance with metrics like IoU or pixel accuracy

Dataset:
- CamVid (mounted from Google Drive)

Tools:
- Google Colab
- PyTorch or TensorFlow
- matplotlib
