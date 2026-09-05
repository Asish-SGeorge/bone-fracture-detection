# Bone Fracture Detection

A Streamlit application that classifies bone X-ray images with a Hugging Face Vision Transformer and displays an approximate model-attention region.

> **Educational use only:** This application is not a medical device and must not replace evaluation by a qualified healthcare professional.

## Features

- Bone-fracture image classification.
- Approximate red attention box and heatmap.
- CPU-compatible setup for local testing.
- Optional training example for a labeled image dataset.

## Run locally

Create or activate the project environment, then install dependencies:

```powershell
py -3.10 -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.\\.venv\\Scripts\\streamlit.exe run app.py
```

Open `http://localhost:8501` and upload a JPG or PNG X-ray image. The first run downloads the model from Hugging Face.

## Training example

`train_example.py` contains a template for fine-tuning a ViT model. Provide a labeled Hugging Face dataset or local image-folder dataset before enabling the dataset-loading section.

## Model and limitations

The app uses `prithivMLmods/Bone-Fracture-Detection`. The displayed box and heatmap are attention estimates, not validated fracture-localization annotations. Do not use the output for diagnosis or treatment decisions.
