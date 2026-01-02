import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model


def build_multi_head_sda(window_size=10, physics_features=3, strategic_features=3):
    # --- Branch 1: The Physics Head (LSTM) ---
    physics_input = Input(shape=(window_size, physics_features), name="Physics_Input")
    x1 = LSTM(64, return_sequences=True)(physics_input)
    x1 = LSTM(32)(x1)
    x1 = Dropout(0.2)(x1)

    # --- Branch 2: The Strategic Head (Dense) ---
    # Strategic data is often less "sequential" and more "contextual"
    strat_input = Input(shape=(strategic_features,), name="Strategic_Input")
    x2 = Dense(32, activation='relu')(strat_input)
    x2 = Dense(16, activation='relu')(x2)

    # --- Fusion: Merging Physics and Strategy ---
    combined = Concatenate()([x1, x2])

    # Final Decision Layers
    z = Dense(32, activation='relu')(combined)
    z = Dense(16, activation='relu')(z)

    # Final Output: Probability (0-1)
    output = Dense(1, activation='sigmoid', name="Maneuver_Probability")(z)

    model = Model(inputs=[physics_input, strat_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


# Initialize the 2026 Fleet Predictor
model = build_multi_head_sda()
model.summary()