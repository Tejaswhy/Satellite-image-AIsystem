# Satellite Image AI System

## Overview

Satellite Image AI System is a machine learning based application designed to detect abnormal patterns in satellite imagery. The system uses deep learning techniques to identify potential land-use anomalies such as deforestation, illegal construction, and river encroachment. The objective of this project is to demonstrate how artificial intelligence can assist in monitoring environmental and urban changes through automated analysis of satellite images.

## Problem Statement

Monitoring large geographical regions manually is extremely difficult and time-consuming. Activities such as illegal deforestation, unauthorized buildings, and encroachment into rivers or lakes often go undetected. Satellite imagery provides valuable information, but analyzing thousands of images manually is inefficient. This project proposes an automated AI-based solution to detect anomalies in satellite images and assist in environmental monitoring.

## System Approach

The system follows a two-stage pipeline.

First, anomaly detection is performed using the PaDiM model. PaDiM learns the normal feature distribution from training images and identifies unusual patterns that deviate from this distribution.

Second, once an anomaly is detected, an image classifier based on EfficientNet-B0 predicts the type of anomaly. The classifier categorizes detected anomalies into predefined classes such as deforestation, illegal building, and river encroachment.

## Features

Detection of abnormal patterns in satellite images
Classification of detected anomalies into specific categories
Visualization of anomaly regions using heatmaps
Interactive web interface for image upload and analysis
Efficient deep learning pipeline for anomaly detection and classification

## Technologies Used

Python programming language
PyTorch deep learning framework
Anomalib library for anomaly detection
TorchVision for pretrained models
Streamlit for the web interface
NumPy for numerical operations
OpenCV for image processing
Matplotlib for visualization

## Project Structure

app.py contains the main Streamlit application
README.md contains project documentation
requirements.txt lists the project dependencies
.gitignore excludes large files such as trained model weights

## How to Run

Clone the repository from GitHub
Install required dependencies using pip install -r requirements.txt
Place the trained model files in the project directory
Run the application using streamlit run app.py

## Applications

Environmental monitoring
Urban planning and infrastructure monitoring
Detection of illegal deforestation
Identification of unauthorized construction
Satellite-based land use analysis

