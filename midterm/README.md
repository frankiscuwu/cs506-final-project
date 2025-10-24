# Midterm Report (10/27)

## Preliminary visualizations

![alt text](./vis1.png)

## Data processing
The images were already augmented and preprocessed in the dataset. The documentation of the dataset describes the augmentation step as using Keras's ``ImageDataGenerator`` with the following parameters:

```
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=10,         
    width_shift_range=0.1,     
    height_shift_range=0.1,    
    shear_range=0.1,           
    zoom_range=0.1,            
    horizontal_flip=True,      
    fill_mode='nearest',       
    brightness_range=[0.2, 1.2]
)
```
The augmentations include:
- Rotation: Up to 10 degrees.
- Width & Height Shift: Up to 10% of the total image size.
- Shearing & Zooming: 10% variation.
- Horizontal Flip: Randomly flips images for additional diversity.
- Brightness Adjustment: Ranges from 0.2 to 1.2 for varying light conditions.

The images were then processed to be consistent 512x512 pixels in size, with files renamed consistently.

### Data split

Following our proposal, we split our data in the following way:

- Training: 70% (3,500 images/class is 14,000 images total)
- Validation: 15% (750 images/class is 3,000 images total) 
- Testing: 15% (750 images/class is 3,000 images total)

<!-- is this true in the new model? -->
~~The 512px x 512px images had already been preprocessed. After splitting the data, we normalised and resized images to 128 x 128 prior to model training.~~

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
