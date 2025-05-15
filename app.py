from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import logging
import pandas as pd
from flask_cors import CORS
import re
import os

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)

# Constants
MODEL_PATH = "assistant_model.h5"
TOKENIZER_PATH = "tokenizer.pkl"
ENCODER_PATH = "label_encoder.pkl"
CSV_PATH = "queries_dataset_grouped_sorted.csv"
MAX_SEQ_LEN = 20
CONFIDENCE_THRESHOLD = 0.4

# Load model and tools
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"{MODEL_PATH} not found.")
    model = load_model(MODEL_PATH)
    logging.info("✅ Model loaded.")
    
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(f"{TOKENIZER_PATH} not found.")
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    logging.info("✅ Tokenizer loaded.")
    
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(f"{ENCODER_PATH} not found.")
    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)
    logging.info("✅ Label encoder loaded.")
    
except Exception as e:
    logging.error(f"❌ Loading model/tokenizers failed: {str(e)}")
    raise

# Load intent map
try:
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"{CSV_PATH} not found.")
    df = pd.read_csv(CSV_PATH)
    df = df[['Grouped_Intent', 'answer']].dropna().drop_duplicates()
    intent_map = df.groupby('Grouped_Intent')['answer'].first().to_dict()
    intent_map = {k.lower().strip(): v for k, v in intent_map.items()}
    logging.info(f"✅ Intent map loaded with {len(intent_map)} entries.")
except Exception as e:
    logging.error(f"❌ Loading CSV failed: {str(e)}")
    raise

# Greeting responses
greeting_responses = {
    "hello": "Hello! How can I assist you today?",
    "hi": "Hi there! Need help with something?",
    "hey": "Hey! What would you like to do today?",
    "how are you": "I'm doing great, thanks! How about you?",
    "good morning": "Good morning! Hope you have a great day!",
    "good evening": "Good evening! What can I help you with?",
    "whats up": "Not much, just here to help you!",
    "yo": "Yo! Ready when you are!",
    "hiya": "Hiya! What's on your mind?",
    "greetings": "Greetings! How may I serve you today?"
}
ASSISTANT_INTRO = "I'm your university voice assistant. How can I help you today?"

# Intent detection function
def get_intent(user_input):
    try:
        user_input_clean = user_input.lower().strip()
        user_input_clean = re.sub(r'[^\w\s]', '', user_input_clean)

        # Manual greeting detection
        for key in greeting_responses:
            if key in user_input_clean:
                return {
                    "intent": "greeting",
                    "confidence": 1.0,
                    "response": greeting_responses[key] + " " + ASSISTANT_INTRO
                }

        # Manual reminder detection
        reminder_keywords = ["add reminder", "set a reminder", "remind me", "create reminder", "schedule reminder"]
        for keyword in reminder_keywords:
            if keyword in user_input_clean:
                return {
                    "intent": "reminder",
                    "confidence": 1.0,
                    "response": "✅ Reminder added successfully!"
                }

        # Model-based prediction
        seq = tokenizer.texts_to_sequences([user_input_clean])
        if not seq or not seq[0]:
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "response": "Sorry, I couldn't understand that. Please rephrase."
            }

        padded = pad_sequences(seq, maxlen=MAX_SEQ_LEN, padding='post')
        prediction = model.predict(padded)
        intent_index = np.argmax(prediction)
        confidence = float(np.max(prediction))
        intent_label = encoder.inverse_transform([intent_index])[0].lower().strip()

        logging.info(f"User Input: {user_input}")
        logging.info(f"Predicted Intent: {intent_label}, Confidence: {confidence:.2f}")

        if confidence >= CONFIDENCE_THRESHOLD:
            response_message = intent_map.get(intent_label, "I'm not sure how to help with that.")
            return {"intent": intent_label, "confidence": confidence, "response": response_message}
        else:
            return {
                "intent": "unknown",
                "confidence": confidence,
                "response": "I'm not sure I understand. Could you rephrase?"
            }

    except Exception as ex:
        logging.error(f"Prediction failed: {str(ex)}")
        return {
            "intent": "error",
            "confidence": 0.0,
            "response": f"An error occurred during prediction: {str(ex)}"
        }

# Routes
@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Voice Assistant API is running."})

@app.route("/predict_intent", methods=["POST"])
def predict_intent():
    try:
        data = request.get_json(force=True)
        if not data or "query" not in data:
            return jsonify({"error": "Missing 'query' in request body"}), 400

        user_input = data["query"]
        logging.info(f"Received query: {user_input}")
        result = get_intent(user_input)
        logging.info(f"Response: {result}")
        return jsonify(result)

    except Exception as e:
        logging.error(f"Exception in /predict_intent: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Entry point
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
