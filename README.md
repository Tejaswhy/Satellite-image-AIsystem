# Satellite Image Anomaly Detection System

## Project Description

This project presents an AI-based system designed to detect abnormal patterns in satellite images. The goal of the system is to automatically identify unusual land-use activities such as deforestation, illegal building construction, and river encroachment. By using deep learning techniques, the system analyzes satellite imagery and highlights regions that differ from normal patterns.

The system combines anomaly detection and image classification to improve the accuracy of detecting land-use changes. It also provides a visual representation of detected anomalies using heatmaps, allowing users to easily identify suspicious regions in the uploaded satellite image.

## Methodology

The system follows a two-stage pipeline for image analysis.

The first stage uses an anomaly detection model called PaDiM. This model learns the normal distribution of satellite image features during training. When a new image is analyzed, the model identifies regions that deviate from the learned normal patterns. These deviations are treated as anomalies.

The second stage uses an image classification model to categorize the detected anomaly. The classifier predicts whether the anomaly corresponds to deforestation, illegal building activity, or river encroachment.

## Technologies Used

Python is used as the primary programming language for building the system.

PyTorch is used as the deep learning framework for model implementation and inference.

TorchVision provides pretrained models and image transformations used in the classification pipeline.

Anomalib is used to implement the PaDiM anomaly detection algorithm.

Streamlit is used to build the interactive web interface that allows users to upload and analyze satellite images.

NumPy is used for numerical operations during image processing.

OpenCV is used for resizing and processing images for visualization.

Matplotlib is used to generate anomaly heatmaps.

Pillow is used for loading and handling image files.

## System Features

Upload satellite images through a web interface
Detect abnormal regions in satellite imagery
Classify detected anomalies into predefined categories
Generate heatmap overlays highlighting anomalous areas
Provide an interactive interface for visualizing results

## Application Areas

Environmental monitoring
Urban development monitoring
Detection of illegal land use
Remote sensing analysis

## Conclusion

This project demonstrates how artificial intelligence and computer vision can assist in analyzing satellite imagery and detecting abnormal land-use patterns efficiently.
