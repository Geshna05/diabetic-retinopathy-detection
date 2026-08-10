# Diabetic Retinopathy Detection System

A deep learning-based application for detecting and classifying diabetic retinopathy from retinal fundus images. The project uses CNN-based transfer learning for classification and provides a Streamlit interface for real-time inference.

## Overview

Diabetic retinopathy is a diabetes-related eye condition that can lead to vision loss if not detected early. This project explores the use of deep learning and image classification to automatically identify the severity of diabetic retinopathy from retinal fundus images.

The system combines a trained CNN-based transfer learning model with a Streamlit application, allowing users to upload a retinal image and obtain a predicted classification.

## Features

- 5-stage diabetic retinopathy classification
- CNN-based transfer learning
- Image preprocessing and optimization
- Real-time inference through a Streamlit interface
- SQLite-based storage for patient information and prediction results
- Simple and interactive web interface

## Classification Stages

The model classifies retinal images into five stages:

| Class | Severity |
|------:|----------|
| 0 | No Diabetic Retinopathy |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative Diabetic Retinopathy |

## Tech Stack

- **Python**
- **TensorFlow / Keras**
- **Convolutional Neural Networks (CNN)**
- **Transfer Learning**
- **Streamlit**
- **SQLite**
- **NumPy**
- **Pandas**
- **OpenCV**

## System Workflow

```text
Retinal Fundus Image
        ↓
Image Preprocessing
        ↓
CNN-Based Transfer Learning Model
        ↓
Feature Extraction
        ↓
Classification
        ↓
Predicted DR Severity
        ↓
SQLite Storage

## Installation

1. Clone the repository

```bash
git clone https://github.com/Geshna-M/diabetic-retinopathy-detection.git
cd diabetic-retinopathy-detection

2. Create a virtual environment

For Windows:

python -m venv venv

Activate the virtual environment:

venv\Scripts\activate

For macOS/Linux:

python3 -m venv venv

Activate the virtual environment:

source venv/bin/activate
3. Install dependencies
pip install -r requirements.txt
Running the Application

After installing the dependencies, run the Streamlit application:

streamlit run app.py

The application will open automatically in your default browser.

If it does not open automatically, open the URL displayed in the terminal. It will typically be:

http://localhost:8501

Upload a retinal fundus image through the application to get the predicted diabetic retinopathy classification.


**One thing:** make sure your actual main file is named `app.py`. If it's named something else, change:

```bash
streamlit run app.py
