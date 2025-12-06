import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf
from sklearn.metrics import confusion_matrix

def interactive_feature_maps(model, image):
    # Extract first conv layer output
    layer = model.layers[1]   # adjust index for first Conv2D
    feature_model = tf.keras.Model(inputs=model.input, outputs=layer.output)

    feature_maps = feature_model.predict(image[None, ...])[0]
    num_filters = feature_maps.shape[-1]

    # Create slider-enabled interactive visualization
    fig = px.imshow(feature_maps[..., 0], binary_string=True)
    fig.update_layout(
        title="Feature Map Explorer",
        xaxis_title="Width",
        yaxis_title="Height",
        sliders=[{
            "steps": [{
                "method": "update",
                "args": [{"z": [feature_maps[..., i]]}],
                "label": f"Filter {i}"
            } for i in range(num_filters)]
        }]
    )
    fig.show()

def plot_training_interactive(history):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=history.history["accuracy"], mode='lines+markers', name='Train Acc'))
    fig.add_trace(go.Scatter(y=history.history["val_accuracy"], mode='lines+markers', name='Val Acc'))
    fig.update_layout(
        title="Interactive Accuracy Plot",
        xaxis_title="Epoch",
        yaxis_title="Accuracy",
        hovermode="x unified"
    )
    fig.show()

def interactive_confusion_matrix(true_labels, pred_labels, class_names):
    cm = confusion_matrix(true_labels, pred_labels)
    fig = px.imshow(
        cm, text_auto=True, color_continuous_scale='Blues',
        labels=dict(x="Predicted", y="Actual"),
        x=class_names, y=class_names
    )
    fig.update_layout(title="Interactive Confusion Matrix")
    fig.show()

