import tensorflow as tf

num_classes = 4  # benign, early, pre, pro
epochs = 50
IMG_SIZE = (128, 128)

def build_and_train_model(train_ds: tf.data.Dataset, val_ds: tf.data.Dataset) -> None:
    model3 = tf.keras.Sequential([
    tf.keras.layers.Input(shape=IMG_SIZE + (3,)),

    tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
    tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
    tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model3.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)

    history3 = model3.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[early_stopping])
    
    model3.save("saved_model.keras")
    print("Model saved to ./saved_model.keras")
    
    return model3, history3
