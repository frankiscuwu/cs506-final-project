from download_dataset import download_dataset
from preprocessing import preprocess
from model import build_and_train_model
from predict import predict

if __name__ == "__main__":
    # Download the dataset
    download_dataset()
    print("Dataset downloaded.")
    
    # Preprocess the data
    train_ds, val_ds, test_ds, class_names = preprocess()
    print("Data preprocessing completed.")
    
    # Build and train the model
    model = build_and_train_model(train_ds, val_ds)
    print("Model training completed.")
    
    # Predict and evaluate
    predict(model, test_ds)