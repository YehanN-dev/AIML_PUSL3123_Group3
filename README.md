# AIML Group 3

This repository contains the Group 3 coursework implementation for the AI and Machine Learning module. The project focuses on gait based user authentication using motion sensor data, feature extraction, dimensionality reduction, and machine learning classification.

## Project Overview

The system uses accelerometer and gyroscope data collected from 10 users. Each user's data is segmented into overlapping windows, converted into numerical features, prepared for training and testing, and then evaluated using user specific classifiers.

The main aim is to identify users based on walking patterns and evaluate authentication performance using metrics such as FAR, FRR, and EER.

## Folder Structure

```text
AIML_Group-3/
│
├── code/
│   ├── extract_features_from_windows.m
│   ├── script_1.m
│   ├── script_2.m
│   └── script_3.m
│
├── data/
│   ├── U1NW_FD.csv
│   ├── U1NW_MD.csv
│   ├── ...
│   └── U10NW_MD.csv
│
├── figures/
│   ├── 01_raw_data_sample.png
│   ├── 02_confusion_matrix.png
│   ├── 03_per_user_accuracy.png
│   ├── 04_roc_curves.png
│   └── roc_curves.png
│
├── results/
│   ├── auth_metrics.csv
│   ├── baseline_results.csv
│   ├── extracted_features.mat
│   ├── prepared_data.mat
│   ├── results.mat
│   └── trained_network.mat
│
└── README.md
