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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)

# Constants
MODEL_PATH = "assistant_model.h5"
TOKENIZER_PATH = "tokenizer.pkl"
ENCODER_PATH = "label_encoder.pkl"
CSV_PATH = "queries_dataset_grouped_sorted.csv"
MAX_SEQ_LEN = 20
CONFIDENCE_THRESHOLD = 0.0

# Globals
model = None
tokenizer = None
encoder = None
intent_map = {}
intent_queries = {}

def load_components():
    global model, tokenizer, encoder, intent_map, intent_queries

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

        if not os.path.exists(CSV_PATH):
            raise FileNotFoundError(f"{CSV_PATH} not found.")
        df = pd.read_csv(CSV_PATH)
        df.columns = df.columns.str.strip()

        df = df[['Grouped_Intent', 'Query', 'answer']].dropna().drop_duplicates()

        # Build intent -> answer map (use first answer for each intent)
        intent_map = df.groupby('Grouped_Intent')['answer'].first().to_dict()
        intent_map = {k.lower().strip(): v for k, v in intent_map.items()}

        # Build intent -> [(question, answer), ...] map for TF-IDF matching
        intent_queries.clear()
        for intent in df['Grouped_Intent'].unique():
            filtered = df[df['Grouped_Intent'] == intent]
            intent_queries[intent.lower()] = list(zip(filtered['Query'].str.lower(), filtered['answer']))

        logging.info(f"✅ Loaded {len(intent_map)} intent responses.")

    except Exception as e:
        logging.error(f"Error loading components: {e}")
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

def get_best_matching_answer(user_input, intent_label):
    if intent_label not in intent_queries:
        return intent_map.get(intent_label, "I'm not sure how to help with that.")

    qa_pairs = intent_queries[intent_label]
    questions, answers = zip(*qa_pairs)

    # Fit vectorizer on known questions + current user input
    vectorizer = TfidfVectorizer()
    vectorizer.fit(list(questions) + [user_input])
    tfidf_matrix = vectorizer.transform(list(questions) + [user_input])

    # Compute cosine similarity between user input (last vector) and all known questions
    cosine_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]

    best_match_index = cosine_scores.argmax()
    best_score = cosine_scores[best_match_index]

    logging.info(f"TF-IDF best match score: {best_score:.2f}")

    if best_score >= 0.3:
        return answers[best_match_index]
    else:
        return intent_map.get(intent_label, "I'm not sure how to help with that.")

def get_intent(user_input):
    try:
        user_input_clean = user_input.lower().strip()
        user_input_clean = re.sub(r'[^\w\s]', '', user_input_clean)

        # Check for greetings first
        for key in greeting_responses:
            if key in user_input_clean:
                return {
                    "intent": "greeting",
                    "confidence": 1.0,
                    "response": greeting_responses[key] + " " + ASSISTANT_INTRO
                }

        # Check for reminder keywords
        reminder_keywords = ["add reminder", "set a reminder", "remind me", "create reminder", "schedule reminder"]
        for keyword in reminder_keywords:
            if keyword in user_input_clean:
                return {
                    "intent": "reminder",
                    "confidence": 1.0,
                    "response": "✅ Reminder added successfully!"
                }

        # Tokenize and pad
        seq = tokenizer.texts_to_sequences([user_input_clean])
        if not seq or not seq[0]:
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "response": "Sorry, I couldn't understand that. Please rephrase."
            }

        padded = pad_sequences(seq, maxlen=MAX_SEQ_LEN, padding='post')
        prediction = model.predict(padded, verbose=0)
        intent_index = np.argmax(prediction)
        confidence = float(np.max(prediction))
        intent_label = encoder.inverse_transform([intent_index])[0].lower().strip()

        logging.info(f"User Input: {user_input}")
        logging.info(f"Predicted Intent: {intent_label}, Confidence: {confidence:.2f}")

        if confidence >= CONFIDENCE_THRESHOLD:
            response_message = get_best_matching_answer(user_input_clean, intent_label)
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


if __name__ == "__main__":
    load_components()
    app.run(host="0.0.0.0", port=5000, debug=True)
