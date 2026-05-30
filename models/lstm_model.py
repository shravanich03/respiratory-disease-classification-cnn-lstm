import tensorflow as tf
from tensorflow.keras import layers, models

def build_lstm_model(input_shape=(128, 40), num_classes=5):
    """
    LSTM model for respiratory disease classification.
    Input: MFCC sequence (128 time steps x 40 coefficients)
    Output: 5-class softmax prediction
    Classes: Normal, Pneumonia, COPD, Asthma, Bronchitis
    """
    model = models.Sequential([

        # LSTM Layer 1 — returns full sequence
        layers.LSTM(128, return_sequences=True, input_shape=input_shape),
        layers.Dropout(0.3),

        # LSTM Layer 2 — returns final hidden state only
        layers.LSTM(64),
        layers.Dropout(0.3),

        # Fully Connected Layer
        layers.Dense(64, activation='relu'),

        # Output Layer
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
