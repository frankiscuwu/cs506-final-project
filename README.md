# Predicting Cancer Diagnoses using Image Classification

This project's goal is creating a model that can identify cancer stage/categorization results based on labelled images. We will focus on acute lymphoblastic leukemia (ALL) microscopic cellular images in our training, identifying benign, early pre-b, pre-b, and pro-b diagnoses.

We collected the data from [Obuli Sai Naren's Multi Cancer Dataset on Kaggle](https://www.kaggle.com/datasets/obulisainaren/multi-cancer/data). This dataset includes categorized and sorted images for ALL, brain, breast, cervical, kidney, lung, colon, lymphoma, and oral cancer, and was augmented using Keras's `ImageDataGenerator` class.

The model will use image processing tools like OpenCV to classify and model the data. To visualize our model's results, we will use bar graphs, ROC curves, and image heatmaps to display accuracy metrics. Our plan will be to utilize a subset of the total dataset in training, before testing it with a separate subset proportional to ALL distribution in real life.