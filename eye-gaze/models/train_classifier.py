import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def build_classifier(input_dim, l2_reg=1e-4):
    model = Sequential([
        Dense(64, input_shape=(input_dim,), activation='relu', kernel_regularizer=l2(l2_reg)),
        Dropout(0.2),
        Dense(32, activation='relu', kernel_regularizer=l2(l2_reg)),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_classifier(X_train, y_train, X_val, y_val, save_path='./outputs/models/nn_classifier.h5'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model = build_classifier(X_train.shape[1])
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=200, batch_size=32, callbacks=[es], verbose=1)
    model.save(save_path)
    return save_path
