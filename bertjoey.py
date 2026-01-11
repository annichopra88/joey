import speech_recognition as sr
import pyttsx3
import time
import random
import re
import joblib
from datetime import datetime
import requests  # For location fetching
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# === Joey's Brain (Intents) ===
intents = {
    "greet": ["hello", "hi", "hey", "good morning", "good evening"],
    "ask for help": ["what can you do", "help me", "how can you assist", "what else can you do"],
    "emergency call": ["call someone", "call emergency", "help me", "emergency", "distress", "urgent"],
    "tell a joke": ["tell me a joke", "make me laugh", "say something funny"],
    "joke feedback": ["not funny", "bad joke", "you are not funny"],
    "thank you": ["thank you", "thanks", "appreciate it"],
    "stop or exit": ["stop", "exit", "quit", "turn off", "bye"],
    "introduce myself": ["my name is", "i am"],
    "ask location": ["where am i", "what city am i in", "my location", "where am i located"],
    "tell time": ["what time is it", "tell me the time", "what's the time"]
}

jokes = [
    "Parallel lines have so much in common… it’s a shame they’ll never meet.",
    "I told my computer I needed a break, and now it won’t stop sending me beach pictures.",
    "Why don’t scientists trust atoms? Because they make up everything."
]

# === Joey's Voice Setup ===
engine = pyttsx3.init()
engine.setProperty('rate', 180)
recognizer = sr.Recognizer()

# === Loading the Finetuned Model and Vectorizer ===
model = joblib.load("C:/Users/ANIRUDH/OneDrive/Desktop/voicebot2/finetuned_model.joblib")
vectorizer = joblib.load("C:/Users/ANIRUDH/OneDrive/Desktop/voicebot2/finetuned_vectorizer.joblib")

# === Speak Function ===
def speak(text):
    print(f"Joey: {text}")
    engine.say(text)
    engine.runAndWait()

# === Match Intent with the Loaded Model ===
def match_intent(user_input):
    user_vec = vectorizer.transform([user_input])
    score = model.predict_proba(user_vec)  # Get probabilities for the model
    best_match_idx = score.argmax()
    best_score = score[0, best_match_idx]
    intent = model.classes_[best_match_idx]
    return intent, best_score

# === Extract Name ===
def extract_name(user_input):
    match = re.search(r"(my name is|i am)\s+([a-zA-Z]+)", user_input.lower())
    if match:
        return match.group(2).capitalize()
    return None

# === Location Function ===
def get_location():
    try:
        response = requests.get("https://ipinfo.io/json")
        data = response.json()
        city = data.get("city", "Unknown City")
        region = data.get("region", "Unknown Region")
        country = data.get("country", "Unknown Country")
        return f"{city}, {region}, {country}"
    except:
        return "Sorry, I couldn’t get your location."

# === Time Function ===
def tell_time():
    current_time = datetime.now().strftime("%H:%M:%S")
    return f"The current time is {current_time}"

# === Simulated Speed + Signal ===
def get_current_speed():
    return random.randint(30, 80)

def get_traffic_signal_status():
    return random.choices(['green', 'red'], weights=[4, 1])[0]

# === Listen Function ===
def listen():
    with sr.Microphone() as source:
        print("Calibrating microphone for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Listening...")
        audio = recognizer.listen(source, phrase_time_limit=6)
        print("Processing...")
        try:
            text = recognizer.recognize_google(audio)
            print(f"You: {text}")
            return text.lower()
        except sr.UnknownValueError:
            return ""
        except sr.RequestError:
            return ""

# === Emergency Handler ===
def handle_emergency():
    speak("Emergency situation detected. Calling emergency services now. Hold tight.")
    print(">>> Dialing 112...")

# === Speed Warning ===
def detect_speeding(speed_limit=60):
    current_speed = get_current_speed()
    print(f"Current Speed: {current_speed} mph")
    if current_speed > speed_limit:
        speak("Warning! You are speeding. Please slow down!")
        return True
    return False

# === Red Light Check (less frequent) ===
def detect_red_light_violation(last_warning_time):
    signal_status = get_traffic_signal_status()
    current_time = time.time()
    if signal_status == 'red' and (current_time - last_warning_time > 15):
        speak("You jumped the red light! Please stop immediately!")
        return current_time
    return last_warning_time

# === Simulate Heartbeat ===
def simulate_heartbeat():
    return random.randint(60, 100)

# === Distress Handler ===
def handle_distress_signal(user_input, score):
    distress_signals = ["help", "emergency", "urgent", "distress"]
    if any(signal in user_input for signal in distress_signals) and score > 0.75:
        speak("It sounds like you're in distress. I will immediately take action.")
        handle_emergency()
        heartbeat = simulate_heartbeat()
        speak(f"Your current heartbeat is {heartbeat} bpm.")
        speak("Anni, are you alright?")
        user_response = listen()
        if "no" in user_response:
            speak("Calm down, Anni. Do you want me to call someone for you?")
            if "yes" in listen():
                handle_emergency()
        else:
            speak("Good to hear you're fine. Let me know if you need anything.")

# === Main Assistant Loop ===
def main():
    speak("Joey is ready.")
    last_red_light_warning_time = 0

    while True:
        user_input = listen()
        if not user_input:
            continue

        intent, score = match_intent(user_input)
        print(f"Matched Intent: {intent} (score: {round(score, 2)})")

        # Handling distress signal (combined from old code)
        handle_distress_signal(user_input, score)
        
        # Other functions for speeding, red light, etc.
        detect_speeding()
        last_red_light_warning_time = detect_red_light_violation(last_red_light_warning_time)

        # Intent responses
        if intent == "greet":
            speak("Hello, welcome to okDriver. This is Joey. How may I assist you today?")
        elif intent == "ask for help":
            speak("I can help you with navigation, emergency calls, jokes, time, location, and more.")
        elif intent == "tell a joke":
            speak(random.choice(jokes))
        elif intent == "joke feedback":
            speak("Ouch, tough crowd... I'll work on that.")
        elif intent == "thank you":
            speak("You're welcome!")
        elif intent == "introduce myself":
            name = extract_name(user_input)
            if name:
                speak(f"Nice to meet you, {name}!")
            else:
                speak("Nice to meet you!")
        elif intent == "ask location":
            speak(f"You're currently in {get_location()}.")
        elif intent == "tell time":
            speak(tell_time())
        elif intent == "stop or exit":
            speak("Turning off now. Thank you for opting okDriver. Stay safe. Goodbye!")
            break
        else:
            speak("Hmm, I’m still learning. Can you say that another way?")

if __name__ == "__main__":
    main()
