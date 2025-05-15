# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 11:29:02 2025

@author: zohaa
"""

import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load components
model = load_model("assistant_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

def predict_intent(user_input):
    seq = tokenizer.texts_to_sequences([user_input])
    padded = pad_sequences(seq, maxlen=20, padding='post')
    pred = model.predict(padded)
    intent_index = np.argmax(pred)
    intent_label = encoder.inverse_transform([intent_index])[0]
    return intent_label

