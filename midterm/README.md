# Midterm Report (10/27)

## Preliminary visualizations

## Data processing
Following our proposal, we split our data in the following way:

- Training: 70% (3,500 images/class)
- Validation: 15% (750 images/class) 
- Testing: 15% (750 images/class)

The 512px x 512px images had already been preprocessed. After splitting the data, we normalised and resized images to 128 x 128 prior to model training. 

## Data modeling methods
We developed a Convolutional Neural Network (CNN) for multi-class image classification using TensorFlow/Keras. The CNN takes the 128×128×3 RGB images (normalized to [0, 1] intensity range) and classfies them into benign, early pre-B, pre-B and pro-B ALL sub-types.

CNN Architecture:
- Conv2D (32 filters, 3×3, ReLU): learns low-level spatial features
- MaxPooling2D (2×2): reduces spatial dimensions, retains key activations
- Conv2D (64 filters, 3×3, ReLU): captures higher-order texture and shape features
- MaxPooling2D (2×2): further down-samples feature maps
- Flatten: converts 3-D feature maps to a 1-D feature vector
- Dense (128 units, ReLU): learns global feature representations for classification
- Dense (4 units, Softmax): outputs class-probability distribution across the four ALL subtypes

## Preliminary results



-   Preliminary visualizations of data.
-   Detailed description of data processing done so far.
-   Detailed description of data modeling methods used so far.
-   Preliminary results. (e.g. we fit a linear model to the data and we achieve promising results, or we did some clustering and we notice a clear pattern in the data)
