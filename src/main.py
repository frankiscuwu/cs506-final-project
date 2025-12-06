from download_dataset import download_dataset
from preprocessing import preprocess
from model import build_and_train_model
from predict import predict
import os
import tensorflow as tf
from visualizations import interactive_feature_maps, plot_training_interactive, interactive_confusion_matrix

if __name__ == "__main__":
    # Download the dataset
    download_dataset()
    print("Dataset downloaded.")
    
    # Preprocess the data
    train_ds, val_ds, test_ds, class_names = preprocess()
    print("Data preprocessing completed.")
    
    # Build and train the model
    if os.path.exists("saved_model"):
        print("Loading saved model...")
        model = tf.keras.models.load_model("saved_model")
        history = None  
    else:
        print("Training model...")
        model, history = build_and_train_model(train_ds, val_ds)
    print("Model training completed.")
    
    # Predict and evaluate
    predict(model, test_ds)
    
    # Render interactive visualizations
    for images, labels in test_ds.take(1):
        sample_image = images[0].numpy()
        interactive_feature_maps(model, sample_image)
    plot_training_interactive(model.history)
    interactive_confusion_matrix(test_ds, model, class_names)