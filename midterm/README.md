# Midterm Report (10/27)

## Preliminary visualizations

![alt text](./vis1.png)

## Data processing
The images were already augmented and preprocessed in the dataset. The documentation of the dataset describes the augmentation step as using Keras's ``ImageDataGenerator`` with the following parameters:

```
from keras.preprocessing.image import ImageDataGenerator

ImageDataGenerator(
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

### Data split and model preparation

We first assigned each category of Acute Lymphoblastic Leukemia (ALL) with an integer value:

``` 
{
    benign: 0,
    early pre-B: 1,
    pre-B: 2,
    pro-B: 3
}
```

We then iterated through the dataset and tagged each image with the appropriate integer label.

With every image labelled, we then randomly shuffled the images into different subsets following our proposal:

- Training: 70% (3,500 images/class is 14,000 images total)
- Validation: 15% (750 images/class is 3,000 images total) 
- Testing: 15% (750 images/class is 3,000 images total)

We put each subset into its own custom ```ALLDatasetSplit``` class that inherits from ```torch.utils.data.Dataset``` for future use. 

Through these steps, the dataset is properly structured, labelled, and randomly distributed for model training and evaluation.

To make the process more efficient and replicable, we utilized the ```torch.utils.data.DataLoader``` to specify batch sizing and minimize overfitting in the the training step.

<!-- is this true in the new model? -->
~~The 512px x 512px images had already been preprocessed. After splitting the data, we normalized and resized images to 128 x 128 prior to model training.~~

## Data modeling methods
We developed a two Convolutional Neural Networks (CNN) for multi-class image classification using Torch, the second being the same as the first but with an additional dropout layer

Second CNN Architecture:

1) Conv Layer 1: 3 input channels → 32 filters (3×3 kernel)

2) Conv Layer 2: 32 input channels → 64 filters (3×3 kernel)

3) Max Pooling: 2×2 pooling reduces spatial dimensions

4) Dropout (p=0.25): Regularizes the network and mitigates overfitting (omit this layer in our first model)

5) Flatten Layer: Converts feature maps to a 1D vector

6) Fully Connected Layer 1: 64 × 62 × 62 input features → 128 hidden units with ReLU activation

7) Fully Connected Layer 2 (Output): 128 input features → 4 output neurons (class logits)

All convolutional and fully connected layers use ReLU activations to introduce non-linearity and improve learning efficiency.

We then created some functions to train the model, including the ```train_epoch``` function, ```validate_epoch``` function, and ```train_model``` function.



<!-- i believe this is the old model? -->
~~We developed a Convolutional Neural Network (CNN) for multi-class image classification using TensorFlow/Keras. The CNN takes the 128×128×3 RGB images (normalized to [0, 1] intensity range) and classifies them into benign, early pre-B, pre-B and pro-B ALL sub-types.~~

~~CNN Architecture:~~
- Conv2D (32 filters, 3×3, ReLU): learns low-level spatial features
- MaxPooling2D (2×2): reduces spatial dimensions, retains key activations
- Conv2D (64 filters, 3×3, ReLU): captures higher-order texture and shape features
- MaxPooling2D (2×2): further down-samples feature maps
- Flatten: converts 3-D feature maps to a 1-D feature vector
- Dense (128 units, ReLU): learns global feature representations for classification
- Dense (4 units, Softmax): outputs class-probability distribution across the four ALL subtypes

## Preliminary results

Our training


-   Preliminary visualizations of data.
-   Detailed description of data processing done so far.
-   Detailed description of data modeling methods used so far.
-   Preliminary results. (e.g. we fit a linear model to the data and we achieve promising results, or we did some clustering and we notice a clear pattern in the data)
