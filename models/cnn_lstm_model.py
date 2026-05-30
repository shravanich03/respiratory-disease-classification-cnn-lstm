import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn_lstm_model(num_slices=16, slice_height=128, slice_width=8, num_classes=5):
    """
    Hybrid CNN-LSTM model for respiratory disease classification.
    Mel-spectrogram (128x128) is divided into 16 time-slices of shape (128 x 8).
    CNN extracts spatial features per slice.
    LSTM models temporal evolution across slices.
    Output: 5-class softmax prediction
    Classes: Normal, Pneumonia, COPD, Asthma, Bronchitis
    """
    input_layer = layers.Input(shape=(num_slices, slice_height, slice_width, 1))

    # TimeDistributed CNN — same CNN applied to each time slice
    x = layers.TimeDistributed(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))(input_layer)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)

    x = layers.TimeDistributed(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)

    x = layers.TimeDistributed(layers.Flatten())(x)

    # LSTM layers — model temporal evolution of spatial features
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.LSTM(64)(x)

    # Dense output
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    output_layer = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=input_layer, outputs=output_layer)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
