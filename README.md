Chest X-Ray Multi-Label Disease Classification (PyTorch)

A deep learning pipeline for multi-label classification of thoracic diseases from chest X-ray images using the NIH ChestX-ray14 dataset. The model predicts the presence of 14 possible diseases from a single radiograph using a ResNet18 CNN backbone trained with PyTorch.

This project demonstrates how to build a real-world medical imaging ML pipeline, including dataset preparation, imbalance handling, model training, threshold tuning, and inference packaging.


PROBLEM

Chest X-rays can contain multiple co-existing conditions.  
This makes the task multi-label classification rather than standard classification.

Each image can have 0 to N diseases among the following:

- Atelectasis
- Cardiomegaly
- Effusion
- Infiltration
- Mass
- Nodule
- Pneumonia
- Pneumothorax
- Consolidation
- Edema
- Emphysema
- Fibrosis
- Pleural_Thickening
- Hernia


KEY FEATURES

Multi-Label CNN Architecture
- Pretrained ResNet18 backbone
- Custom 14-output classification head

Class Imbalance Handling
Medical datasets are highly imbalanced.

This project uses:
- Weighted BCEWithLogitsLoss
- Per-class positive weighting

Threshold Optimization
Instead of using a global probability threshold (0.5), the pipeline performs:
- Per-class threshold tuning
- Validation-based optimization
- Improved macro F1 score

Efficient Training
- GPU support
- Dataset subsampling for experimentation
- Data augmentation
- Learning rate scheduler


PROJECT STRUCTURE

.
├── train.py or notebook
├── test.py
├── model.pth
└── README.md


MODEL ARCHITECTURE

Backbone:
ResNet18 (ImageNet pretrained)

Final layer:
Linear(512 → 14)

Loss function:
BCEWithLogitsLoss (with class weights)

Optimizer:
Adam

Scheduler:
ReduceLROnPlateau


DATASET

Dataset used:
NIH ChestX-ray14

- ~100k frontal chest X-ray images
- 14 disease labels
- Multi-label classification task

This implementation uses the 224×224 resized version of the dataset from Kaggle.

Dataset source:
https://www.kaggle.com/datasets/khanfashee/nih-chest-x-ray-14-224x224-resized


INSTALLATION

Install dependencies:

pip install torch torchvision numpy pandas scikit-learn pillow

Optional (for dataset download):

pip install kaggle


TRAINING

Run the training pipeline:

python train.py

Training includes:
1. Dataset preprocessing
2. Multi-label encoding
3. Train / validation split
4. Model training
5. Validation evaluation
6. Threshold tuning
7. Checkpoint export

Example training output:

Epoch 9/10
Train Loss: 0.597
Val Macro-F1: 0.179

After threshold tuning:

Macro-F1: 0.239


INFERENCE

To generate predictions on a folder of images:

python test.py <input_image_directory>

Example:

python test.py ./test_images

Output file:

predictions.csv

Format:

filename,Atelectasis,Cardiomegaly,...,Hernia
img_1.png,0,0,1,...,0
img_2.png,1,0,0,...,0


PERFORMANCE

Validation results:

Macro F1 (baseline threshold) : 0.179  
Macro F1 (optimized thresholds) : 0.239


FUTURE IMPROVEMENTS

Potential improvements include:

- DenseNet121 (commonly used for medical imaging)
- Vision Transformers
- Focal Loss
- Stronger augmentation
- Label correlation modeling
- Grad-CAM explainability


TECH STACK

Python  
PyTorch  
Torchvision  
NumPy  
Pandas  
Scikit-learn  
PIL


WHY THIS PROJECT MATTERS

Medical imaging models must deal with:

- Severe class imbalance
- Multi-label prediction
- High-dimensional image inputs
- Strict evaluation metrics

This project demonstrates how to build a practical deep learning pipeline for healthcare imaging tasks using PyTorch.
