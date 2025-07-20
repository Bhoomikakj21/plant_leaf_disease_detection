#!/bin/bash

echo "Step 1: Installing required packages..."
pip install -r ../requirements.txt
pip install jupyter nbconvert

echo "Step 2: Executing Jupyter Notebook: code.ipynb"
KERNEL=$(jupyter kernelspec list | grep -m1 python | awk '{print $1}')
jupyter nbconvert --to notebook --execute ./code.ipynb --output executed_code.ipynb --ExecutePreprocessor.kernel_name=$KERNEL


echo "Step 3: Checking required files for Streamlit app..."

APP_DIR="Disease classification app"

if [ ! -f "$APP_DIR/class_indices.json" ]; then
    echo "Error: class_indices.json not found in $APP_DIR"
    exit 1
fi

if [ ! -d "$APP_DIR/trained_model" ]; then
    echo "Error: trained_model folder not found in $APP_DIR"
    exit 1
fi

echo "All required files found."

echo "Step 4: Launching the Streamlit Plant Disease Detection App..."
cd "$APP_DIR"
streamlit run app.py
