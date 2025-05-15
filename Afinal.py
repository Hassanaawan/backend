# -*- coding: utf-8 -*-
"""
Voice Assistant for University Students
ML-based intent prediction + dataset answer matching
Listens after 'Hey Assistant', exits on 'stop' (real wake word)
@author: zohaa
"""

import spacy
import pyttsx3
import speech_recognition as sr
import pyjokes
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import dateparser
import pickle
import numpy as np
import pandas as pd
import pygame
import tempfile
from gtts import gTTS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load ML model and encoders
model = load_model("assistant_model.h5")
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)
with open("label_encoder.pkl", "rb") as enc:
    lbl_encoder = pickle.load(enc)

# Load dataset
dataset = pd.read_csv("queries_dataset_grouped_sorted.csv")
dataset['Query'] = dataset['Query'].astype(str)

# Config
MAX_LEN = 20
nlp = spacy.load("en_core_web_sm")

# New speak function using gTTS + pygame
def speak(text):
    try:
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            temp_path = fp.name
            tts.save(temp_path)

        pygame.mixer.init()
        pygame.mixer.music.load(temp_path)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            continue
    except Exception as e:
        print("Speech error:", e)

def recognize_voice():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        try:
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)
            print("Recognizing...")
            command = recognizer.recognize_google(audio)
            print(f"Command: {command}")
            return command
        except sr.WaitTimeoutError:
            speak("Listening timed out.")
        except sr.UnknownValueError:
            speak("Sorry, I did not understand that.")
        except sr.RequestError:
            speak("Sorry, my speech service is down.")
    return None

def predict_intent(query):
    seq = tokenizer.texts_to_sequences([query])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    pred = model.predict(padded)
    intent_index = np.argmax(pred)
    confidence = np.max(pred)
    intent = lbl_encoder.inverse_transform([intent_index])[0]
    print(f"Predicted Intent: {intent} | Confidence: {round(confidence, 2)}")
    return intent

def handle_intent(intent, user_input):
    small_talk = {
        "greeting": "Hello! How can I assist you today?",
        "joke": pyjokes.get_joke(),
        "time": f"The current time is {datetime.now().strftime('%I:%M %p')}.",
        "goodbye": "Goodbye! Have a great day."
    }

    if intent in small_talk:
        return small_talk[intent]

    subset = dataset[dataset['Grouped_Intent'] == intent]
    if subset.empty:
        return "I'm not sure how to help with that."

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([user_input] + subset['Query'].tolist())
    similarity = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    best_match_index = similarity.argmax()

    if similarity[best_match_index] < 0.3:
        return "I'm not sure how to help with that."

    return subset.iloc[best_match_index]['answer']

# In-memory reminder store
reminder_list = []

# Background task to check reminders every minute
def check_reminders():
    now = datetime.now()
    for reminder in reminder_list[:]:
        if now >= reminder["time"]:
            speak(f"Reminder: {reminder['task']}")
            print(f"Reminder: {reminder['task']}")
            reminder_list.remove(reminder)

scheduler = BackgroundScheduler()
scheduler.add_job(check_reminders, 'interval', seconds=60)
scheduler.start()

def add_reminder(task):
    speak("When would you like to be reminded?")
    time = recognize_voice()
    if not time:
        speak("Sorry, I couldn't understand the time. Please say it again.")
        return "I couldn't set the reminder."

    # Parse the time using dateparser
    date_time = dateparser.parse(time, settings={'PREFER_DATES_FROM': 'future'})
    if not date_time:
        return "I couldn't understand the reminder time. Please say it again more clearly."

    reminder_list.append({"task": task, "time": date_time})
    return f"Reminder set for {date_time.strftime('%A %I:%M %p')}: {task}"

def main():
    speak("Welcome to the university voice assistant. Say 'Hey Assistant' to activate or 'stop' to exit.")
    active = False

    while True:
        command = recognize_voice()
        if not command:
            continue

        command = command.lower().strip()

        intent = None  # Initialize intent here to avoid UnboundLocalError

        if not active:
            if "hey assistant" in command:
                speak("Assistant activated. I'm now listening for your questions.")
                active = True
            elif "stop" in command:
                speak("Goodbye! Shutting down.")
                break
            else:
                continue
        else:
            if "stop" in command:
                speak("Goodbye! Shutting down.")
                break

            # Check if the user wants to set a reminder
            if "set a reminder" in command:
                speak("What task would you like to be reminded of?")
                task = recognize_voice()
                if not task:
                    speak("Sorry, I couldn't understand the task. Please say it again.")
                    continue

                # Proceed to add the reminder
                response = add_reminder(task)
                speak(response)

            else:
                intent = predict_intent(command)  # This will only be set if there is a valid intent
                if intent:
                    response = handle_intent(intent, command)
                    print(f"Response: {response}")
                    speak(response)

            # Always check if the intent is "goodbye" after processing the command
            if intent == "goodbye":
                break

if __name__ == "__main__":
    main()
