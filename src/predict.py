import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

def predict(model, dataset):
    test_ds_eval = dataset.cache().prefetch(0)
    pred_probs = model.predict(test_ds_eval)
    predicted_labels = np.argmax(pred_probs, axis=1)
