import tensorflow as tf
import numpy as np

BATCH_SIZE = 32
IMG_SIZE = (128, 128)
SEED = 123

def preprocess() -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list[str]]:
    full_ds = tf.keras.utils.image_dataset_from_directory(
        "data/raw/ALL",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED
    )

    # Get class names from the full dataset before splitting
    class_names = full_ds.class_names
    num_classes = len(class_names)
    print("\nClasses:", class_names)

    dataset_size = tf.data.experimental.cardinality(full_ds).numpy()

    train_size = int(0.7 * dataset_size)
    val_size   = int(0.15 * dataset_size)
    test_size  = dataset_size - train_size - val_size  # ensures exact total

    train_ds = full_ds.take(train_size)
    temp_ds  = full_ds.skip(train_size)

    val_ds = temp_ds.take(val_size)
    test_ds = temp_ds.skip(val_size)

    train_ds = train_ds.shuffle(5000)

    print("\nTrain batches:", tf.data.experimental.cardinality(train_ds).numpy())
    print("Val batches:", tf.data.experimental.cardinality(val_ds).numpy())
    print("Test batches:", tf.data.experimental.cardinality(test_ds).numpy())

    for i, class_name in enumerate(class_names):
        print(f"Label {i}: {class_name}")

    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        labels = []
        for _, y in ds:
            labels.extend(y.numpy())
        print(name, np.bincount(labels))
        
    # normalize the data so that values are [0,1] instead of [0,255]
    # improves convergence speed and ensures consistency among all features
    normalization_layer = tf.keras.layers.Rescaling(1./255)

    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

    for images, labels in train_ds.take(1):
        print("Images shape:", images.shape)
        print("Labels:", labels.numpy()[:50])

    unique, counts = np.unique(labels.numpy(), return_counts=True)
    print(dict(zip(unique, counts)))

    # BEFORE normalization
    raw_ds = tf.keras.utils.image_dataset_from_directory(
        "data/raw/ALL",
        shuffle=False,
        image_size=(128, 128),
        batch_size=32
    )

    for imgs, labs in raw_ds.take(1):
        print("RAW range:", imgs.numpy().min(), imgs.numpy().max())
        break

    # AFTER normalization (your actual dataset)
    for imgs, labs in train_ds.take(1):
        print("AFTER range:", imgs.numpy().min(), imgs.numpy().max())
        break
    
    return train_ds, val_ds, test_ds, class_names
