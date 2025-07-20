UBID Final Project - Plant Leaf Disease Detection Using Deep Learning

Project Summary:
This project implements a deep learning-based system to detect plant leaf diseases from images. 
Use an ensemble of three CNN models: Xception, DenseNet121, and DeepPlantNet.

Folder Structure:
UBID final project/
├── requirements.txt
├── readme.txt
├── Report/
│   └── UBID final report.pdf
├── Video/
│   └── video.mp4
└── Code/
    ├── run.sh
    ├── code.ipynb
    └── Disease classification app/
        ├── app.py
        ├── class_indices.json
        └── trained_model/
            ├── xception_model.h5
            ├── densenet_model.h5
            └── deepplantnet_model.h5

External Downloads:
- Dataset: https://drive.google.com/file/d/1rki2yvN17l423dcNZVxSy3e3IES5oJp9/view?usp=sharing
    → Extract into: Code/Plant Disease dataset/
- Trained Models: https://drive.google.com/drive/folders/1ysup6kl8CEbeKiFmufGwWYOF66ZrzEHr?usp=sharing
    → Place into: Code/Disease classification app/trained_model/

How to Run:
1. Unzip the project and navigate into the 'Code/' folder.
2. Make the run script executable (only once):
   chmod +x run.sh
3. Run the project:
   ./run.sh

This will:
- Install dependencies
- Execute code.ipynb
- Launch the Streamlit app (app.py)

Requirements:
- Python 
- Packages listed in requirements.txt 
