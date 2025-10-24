# Predicting Cancer Diagnoses using Image Classification

## Goal

This project's goal is creating a model that can identify cancer stage/categorization results based on labelled images. We will focus on acute lymphoblastic leukemia (ALL) microscopic cellular images in our training, identifying benign, early pre-b, pre-b, and pro-b diagnoses. We hope to achieve high sensitivity to minimize false negatives.

## Data

We collected the data from [Obuli Sai Naren's Multi Cancer Dataset on Kaggle](https://www.kaggle.com/datasets/obulisainaren/multi-cancer/data). This dataset includes categorized and sorted images for ALL, brain, breast, cervical, kidney, lung, colon, lymphoma, and oral cancer, and was augmented using Keras's  class, with each cancer region having 5000 images. This is more than enough for both training and testing. Should we want additional images and a more verbose testing/modeling, we could always pull images elsewhere as well, since medical scans should be consistent.

### Data split

Our plan will be to utilize a subset of the total dataset in training, before testing it with a separate subset proportional to ALL distribution in real life, with a separate subset of images for validation in cross-validating.

- Training: 70% (3,500 images/class)
- Validation: 15% (750 images/class) 
- Testing: 15% (750 images/class)

## Model

The model will use image processing tools like OpenCV to classify and model the data. The images are already pretty well preprocessed, looking very similar to each other with both sizing and coloring, thus we expect minimal cleaning/preprocessing of the data. The exact details of this process will be developed upon in the future as we learn more, including the algorithm and model we'll use. 

## Visualizations

We plan to visualize our project through a variety of methods as listed below, which we may change as time goes on.

### Data Exploration Visualizations:
1. **Class Distribution Analysis**: Bar charts showing sample counts per ALL subtype
2. **Image Sample Grid**: Representative images from each category with annotations
3. **Pixel Intensity Distributions**: Histograms comparing intensity patterns across categories
4. **Image Statistics**: Mean, standard deviation, and channel distributions

### Feature Analysis Visualizations:
1. **Feature Extraction Visualization**: 
   - HOG feature visualizations overlaid on original images
   - Texture pattern heatmaps using GLCM features
   - Principal Component Analysis (PCA) plots of extracted features
2. **Feature Importance**: Bar charts showing most discriminative features for classification
3. **Correlation Matrices**: Feature correlation heatmaps to identify redundant features

### Model Performance Visualizations:
1. **Training Progress**: Loss and accuracy curves during model training
2. **Confusion Matrices**: Detailed breakdown of predictions vs. actual classifications
3. **ROC Curves**: Multi-class ROC analysis with Area Under Curve (AUC) metrics
4. **Precision-Recall Curves**: Especially important for medical diagnosis applications
5. **Grad-CAM Heatmaps**: Visual explanations showing which image regions influenced model decisions
6. **Classification Reports**: Precision, recall, F1-score breakdown by class

### Clinical Interpretation Visualizations:
1. **Sensitivity/Specificity Analysis**: Medical-focused metrics visualization
2. **Prediction Confidence Distributions**: Histogram of model confidence scores
3. **Error Analysis**: Visual analysis of misclassified cases to identify pattern improvements

## Plan

### Evaluation Methodology:
Our testing strategy prioritizes medical diagnostic standards:

**Primary Metrics (Medical Focus):**
- **Sensitivity (Recall)**: Minimize false negatives for cancer detection (target >95%)
- **Specificity**: Minimize false positives to reduce unnecessary interventions (target >90%)
- **Positive Predictive Value (PPV)**: Accuracy of positive cancer predictions
- **Negative Predictive Value (NPV)**: Accuracy of negative cancer predictions

**Secondary Metrics:**
- **F1-Score**: Harmonic mean of precision and recall (especially important for imbalanced classes)
- **AUC-ROC**: Area under receiver operating characteristic curve
- **AUC-PR**: Area under precision-recall curve (better for imbalanced datasets)
- **Overall Accuracy**: General model performance measure

### Testing Phases:

**Phase 1: Cross-Validation Testing**
- 5-fold stratified cross-validation on training data
- Hyperparameter optimization and model selection
- Statistical significance testing of results

**Phase 2: Hold-out Testing**
- Final model evaluation on unseen test set (20% of data)
- Comparison between multiple model approaches
- Statistical analysis of performance differences

**Phase 3: Robustness Testing**
- Test model performance on edge cases and low-quality images
- Evaluate computational performance (inference time, memory usage)
- Assess model consistency across different image acquisition conditions

### Validation Strategy:
- **Stratified Sampling**: Ensure proportional representation of all ALL subtypes
- **Real-world Distribution**: Test set composition reflects actual clinical prevalence
- **Clinical Validation**: Compare model predictions with expert pathologist annotations (if available)
- **Computational Benchmarking**: Measure inference time and resource requirements for clinical deployment feasibility

## Citations

Obuli Sai Naren. (2022). Multi Cancer Dataset [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/3415848
