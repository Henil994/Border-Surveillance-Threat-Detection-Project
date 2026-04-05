# AI Acoustic Threat Detection System

## Overview
Hybrid deep learning model using CNN + ViT + LSTM for audio classification.

## Classes
- Human Intrusion
- Mechanical / Drone Threat
- Animal Threat
- Environmental Noise

## Pipeline
Audio → Mel Spectrogram → CNN → ViT → LSTM → Classification

## Setup
pip install -r requirements.txt

## Run
python core/precompute_features.py  
python -m core.train_model 
python main.py 
python evaluate_model.py