# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 17:00:27 2025

@author: zohaa
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import pickle

# Load dataset
df = pd.read_csv("queries_dataset_grouped_sorted.csv")
df = df[['Query', 'Grouped_Intent']].drop_duplicates().dropna()

# Tokenization
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
df['Query'] = df['Query'].astype(str)
tokenizer.fit_on_texts(df['Query'])
X = tokenizer.texts_to_sequences(df['Query'])
X = pad_sequences(X, maxlen=20, padding='post')

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(df['Grouped_Intent'])
y_cat = tf.keras.utils.to_categorical(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# Handle class imbalance
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y),
    y=y
)
class_weights = dict(enumerate(class_weights))

# ✅ EarlyStopping
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# ✅ Define model with Dropout & L2 Regularization
from tensorflow.keras import regularizers
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=5000, output_dim=128, input_length=20),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.5),  # Increased Dropout
    tf.keras.layers.Dense(y_cat.shape[1], activation='softmax')
])

# Compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train
model.fit(
    X_train, y_train,
    epochs=12,
    batch_size=16,
    validation_data=(X_test, y_test),
    class_weight=class_weights,
    callbacks=[early_stop]
)

# Save model and encoders
model.save("assistant_model.h5")
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

# Test query
custom_query = ["Where can I find the fee structure?"]
seq = tokenizer.texts_to_sequences(custom_query)
padded = pad_sequences(seq, maxlen=20, padding='post')
prediction = model.predict(padded)
predicted_intent = encoder.inverse_transform([np.argmax(prediction)])
print("Predicted Intent:", predicted_intent[0])
