# === Required Libraries ===
import speech_recognition as sr
import pyttsx3
import time
import random
import re
import os
import pygame
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from deep_translator import GoogleTranslator
from gtts import gTTS
from langdetect import detect, LangDetectException

# === Initialize Pygame Mixer (for gTTS audio playback) ===
# Initialize only once at the start
try:
    pygame.mixer.init()
    print("[INFO] Pygame mixer initialized.")
    mixer_initialized = True
except Exception as e:
    print(f"[ERROR] Could not initialize pygame mixer: {e}")
    print("Audio playback for languages other than English will not work.")
    mixer_initialized = False


# Dictionary to map language names (and common variations) to codes
# This is used for language mode setting and explicit translation requests
LANGUAGE_CODES = {
    'english': 'en', 'en': 'en', 'default': 'en', 'normal': 'en',
    'hindi': 'hi', 'hi': 'hi', 'हिंदी': 'hi',
    'spanish': 'es', 'es': 'es', 'español': 'es', 'स्पेनिश': 'es',
    'urdu': 'ur', 'ur': 'ur', 'اردو': 'ur', 'उर्दू': 'ur',
    'bangla': 'bn', 'bn': 'bn', 'bengali': 'bn', 'বাংলা': 'bn',
    'japanese': 'ja', 'ja': 'ja', 'जापानी': 'ja', 'জাপানি': 'ja', 'جاپانی': 'ja',
    'german': 'de', 'de': 'de', 'जर्मन': 'de', 'জার্মান': 'de', 'جرمن': 'de',
    'french': 'fr', 'fr': 'fr', 'फरांसीसी': 'fr', 'ফরাসি': 'fr', 'فرانسیسی': 'fr',
    'chinese': 'zh-CN', 'zh-CN': 'zh-CN', 'चीनी': 'zh-CN', 'চীনা': 'zh-CN', 'چینی': 'zh-CN', # Using a specific variant for clarity
    'russian': 'ru', 'ru': 'ru', 'रूसी': 'ru', 'রুশ': 'ru', 'روسی': 'ru',
    'arabic': 'ar', 'ar': 'ar', 'अरबी': 'ar', 'आरबी': 'ar', 'عربی': 'ar',
    # Add more languages here as needed, using ISO 639-1 codes or common variants
    'portuguese': 'pt', 'pt': 'pt',
    'italian': 'it', 'it': 'it',
    'korean': 'ko', 'ko': 'ko',
    'dutch': 'nl', 'nl': 'nl',
}


# === Joey's Brain (Intents) ===
# Expanded intents and phrases. Note: Intent phrases should be in English primarily for TF-IDF matching
# based on the current structure. Translations are handled in responses.
intents = {
    "greet": [
        "hello", "hi", "hey", "good morning", "good evening", "what's up", "yo", "greetings", "howdy",
        "namaste", "salaam", "hola", "konnichiwa", "guten tag", "bonjour", "ciao", "good afternoon"
    ],
    "greet_someone": [ # This intent will be used for generic "greet someone" after specific patterns are checked
        "say hello to someone", "greet someone", "tell someone hello"
    ],
    "ask_for_help": [
        "what can you do", "help me", "how can you assist", "what else can you do", "what are your features", "capabilities", "tell me your functions",
        "what are your abilities", "what features do you have"
    ],
    "emergency_call": [
        "call someone", "call emergency", "help me", "emergency", "distress", "urgent", "call police", "i need help", "sos", "danger", "call an ambulance", "medical emergency", "fire emergency",
        "there is an emergency"
    ],
    "tell_a_joke": [
        "tell me a joke", "make me laugh", "say something funny", "give me a joke", "do you know any jokes", "joke please", "tell joke", "say a joke"
    ],
    "joke_feedback_negative": [
        "not funny", "bad joke", "you are not funny", "that was terrible", "lame joke", "didn't like it", "that wasn't funny"
    ],
    "thank_you": [
        "thank you", "thanks", "appreciate it", "grateful", "thanks a lot", "thank you so much", "thank you joey", "thanks joey"
    ],
    "stop_or_exit": [
        "stop", "exit", "quit", "turn off", "bye", "shut down", "goodbye", "see you later", "terminate", "cancel", "close", "end",
        "stop it now", "alright stop it now", "joey stop", "joey exit", "end program", "close program", "shut down now", "exit program",
        "stop the program", "exit the program", "turn joey off", "alright stop" # Added more variations
    ],
    "introduce_myself": [
        "my name is", "i am", "i'm", "call me", "you can call me"
    ],
    "ask_location": [
        "where am i", "what city am i in", "my location", "where am i located", "current location", "tell me my location"
    ],
    "tell_time": [
        "what time is it", "tell me the time", "what's the time", "current time", "time please", "time now"
    ],
    "ask_weather": [
        "what's the weather", "tell me the weather", "how is the weather", "weather now", "weather forecast", "temperature", "is it raining", "is it sunny", "what's the temperature"
    ],
    "translate": [
        "say this in", "translate this to", "convert to", "how do you say in", "translate to" # Keep these general phrases
    ],
    "ask_name": [
        "what's my name", "who am i", "my name", "do you know my name", "tell me my name"
    ],
    "set_language_mode": [ # These phrases should be used for mode switching via regex
        "hindi on", "hindi mode on", "switch to hindi", "use hindi", "hindi mode",
        "hindi off", "hindi mode off", "stop hindi", "english mode", "default mode", "normal mode",
        "spanish on", "spanish mode on", "switch to spanish", "use spanish", "spanish mode",
        "spanish off", "spanish mode off", "stop spanish",
        "urdu on", "urdu mode on", "switch to urdu", "use urdu", "urdu mode",
        "urdu off", "urdu mode off", "stop urdu",
        "bangla on", "bangla mode on", "switch to bangla", "use bangla", "bangla mode", "bengali on", "bengali mode on", "switch to bengali", "use bengali", "bengali mode",
        "bangla off", "bangla mode off", "stop bangla", "bengali off", "bengali mode off", "stop bengali",
        "default language", "english on", "english mode on", "speak in english",
        # Add phrases for other languages in LANGUAGE_CODES if needed for mode switching
        "japanese mode on", "switch to japanese", "speak in japanese",
        "german mode on", "switch to german", "speak in german",
        # ... etc.
    ],
    "about_joey": [ # Phrases asking about Joey itself
        "what is your name", "who are you", "tell me about yourself", "what are you", "what do you do",
        "your name", "your colour", "your age", "do you have hair", "what color is your hair", "what colour is your hair",
        "do you not have hair", "you don't have hair", "what languages can you speak", "speak any language", "what languages do you know",
        "are you real", "are you alive", "do you have feelings", "are you a robot", "are you human",
        "tell me about you"
    ],
     # === Add Other New Intents Here (e.g., Play Music, Set Reminder) ===
    "play_music": [
        "play a song", "play some music", "put on some tunes", "play music", "play something"
    ],
    "set_reminder": [
        "remind me to", "set a reminder for", "create a reminder", "remind me in", "set reminder to", "add a reminder for"
    ],
    "search_web": [
        "search for", "find me information about", "google", "look up", "search the web for", "search online for"
    ],
    "check_heartbeat": [
        "what's my heartbeat", "check my pulse", "my heart rate", "how is my heartbeat", "check my heartbeat"
    ]
}

# === Multilingual Jokes ===
jokes_multi = {
    'en': [
        "Why don't scientists trust atoms? Because they make up everything!",
        "What do you call a lazy kangaroo? Pouch potato!",
        "Why did the scarecrow win an award? Because he was outstanding in his field!",
        "What concert costs just 45 cents? 50 Cent featuring Nickelback!",
        "Why did the bicycle fall over? Because it was two tired!",
        "What do you call a fish wearing a bowtie? Sofishticated!",
        "Why don't eggs tell jokes? They'd crack each other up!",
        "What did the left eye say to the right eye? Between you and me, something smells!",
        "Why was the math book sad? Because it had too many problems.",
        "What do you call a fake noodle? An impasta!"
    ],
    # Add jokes in other languages here if you have them
    # Example structure:
    # 'es': [
    #     "Joke 1 in Spanish",
    #     "Joke 2 in Spanish"
    # ],
    # 'ja': [
    #     "Joke 1 in Japanese",
    #     "Joke 2 in Japanese"
    # ]
}


# === Speech Engine Setup ===
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('rate', 180)
selected_voice_id = None
try:
    # Prefer a male or female English voice if available
    for voice in voices:
        if 'english' in voice.name.lower() and ('zira' in voice.name.lower() or 'david' in voice.name.lower() or 'mark' in voice.name.lower()):
            selected_voice_id = voice.id
            break
    # Fallback to any English voice
    if not selected_voice_id:
        for voice in voices:
            if 'english' in voice.name.lower():
                selected_voice_id = voice.id
                break
    if selected_voice_id:
        engine.setProperty('voice', selected_voice_id)
        print(f"[INFO] Selected pyttsx3 voice: {engine.getProperty('voice').name}")
    else:
        print("[WARNING] Could not find a preferred English TTS voice. Using default.")
except Exception as e:
    print(f"[WARNING] Error setting pyttsx3 voice: {e}. Using default.")

recognizer = sr.Recognizer()
user_name = None # Variable to store the user's name
active_language_mode = None # Stores the language code ('en', 'hi', 'es', etc.)

# --- Driving Assistance Placeholders ---
last_speeding_warning_time = 0
last_red_light_warning_time = 0
speed_check_interval = 120 # Check speed every 120 seconds
traffic_check_interval = 120 # Check traffic light status every 120 seconds
last_speed_check = time.time()
last_traffic_check = time.time()

# === Intent Recognition Setup (Initialize and Fit TF-IDF) ===
vectorizer = TfidfVectorizer()
intent_phrases = []
intent_tags = []
for tag, phrases in intents.items():
    # Add phrases only in English for TF-IDF training
    if tag != "set_language_mode": # Exclude mode phrases from TF-IDF as they are primarily regex handled
         intent_phrases.extend(phrases)
         intent_tags.extend([tag] * len(phrases))
    else:
         # For language mode intent, add some general phrases for TF-IDF fallback
         intent_phrases.extend(["change language", "switch language", "set language"])
         intent_tags.extend([tag] * 3)


# Fit the vectorizer with all phrases BEFORE the main loop
try:
    X = vectorizer.fit_transform(intent_phrases)
    print("[INFO] TF-IDF vectorizer fitted successfully.")
except Exception as e:
    print(f"[ERROR] Failed to fit TF-IDF vectorizer: {e}")
    print("Intent matching may not work correctly.")


# === Core Functions ===

def get_language_code(lang_name_or_code):
    """Maps a language name or code to a standardized code from LANGUAGE_CODES."""
    # Handle None or empty input
    if not lang_name_or_code:
        return None
    # First check if the input is already a valid code in our list
    if str(lang_name_or_code).lower() in LANGUAGE_CODES.values():
         return str(lang_name_or_code).lower()
    # Then check if it's a language name (key) in our dictionary
    return LANGUAGE_CODES.get(str(lang_name_or_code).lower(), None) # Ensure input is string


def speak(text, lang='en'):
    """Speaks the given text using TTS."""
    # Ensure lang is a valid code, default to 'en' if not found in LANGUAGE_CODES
    lang_code = get_language_code(lang) # Get the standardized code
    if not lang_code:
        print(f"[Speak Info] Language '{lang}' not recognized or supported, using English.")
        lang_code = 'en' # Fallback to English code

    print(f"Joey ({lang_code}): {text}")

    # Use pyttsx3 for English, gTTS for others if mixer is initialized
    if lang_code == 'en':
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
             print(f"[pyttsx3 Speak Error]: {e}")
             # Fallback to print if pyttsx3 fails
             print(f"[Fallback Print - en]: {text}")

    elif mixer_initialized: # Use gTTS only if mixer is initialized
         play_gtts_audio(text, lang_code)
    else:
         print("[WARNING] pygame mixer not initialized. Cannot play non-English audio.")
         print(f"[Fallback Print - {lang_code}]: {text}")
         # speak("Sorry, I cannot speak in that language right now.", 'en') # Avoid recursion


def play_gtts_audio(text, lang_code):
    """Generates audio using gTTS, saves temporarily, plays it, and cleans up."""
    file_path = None # Initialize file_path outside try
    try:
        # Removed the gTTS.get_langs() check due to the error.
        # Relying on the gTTS constructor to raise an error if the language is unsupported.

        tts = gTTS(text=text, lang=lang_code, slow=False)
        # Use a robust temporary file handling
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
             file_path = fp.name
             tts.save(file_path)

        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        pygame.mixer.music.unload()

    except Exception as e: # Catch any exception from gTTS or playback
        print(f"[gTTS/Playback Error - {lang_code}]: {e}")
        # Provide a fallback message in English using pyttsx3
        try:
             fallback_msg = "Sorry, I couldn't generate or play the audio response in that language."
             print(f"Joey (en - Fallback): {fallback_msg}")
             engine.say(fallback_msg)
             engine.runAndWait()
        except Exception as fb_e:
             print(f"[Playback Fallback Error]: {fb_e}")
    finally:
        # Ensure the temporary file is removed
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as cleanup_e:
                print(f"[Audio Cleanup Error] {cleanup_e}")


def translate_text(text, target_lang_code):
    """Translates text to the target language code."""
    try:
        # The deep_translator library often uses ISO 639-1 codes
        # Removed the explicit check using get_supported_languages due to the error.
        # Relying on GoogleTranslator to handle unsupported codes and raise errors.

        translated_text = GoogleTranslator(source='auto', target=target_lang_code).translate(text)
        print(f"Joey (Translated to {target_lang_code}): {translated_text}")
        return translated_text
    except Exception as e:
        print(f"[Translation Error to {target_lang_code}]: {e}")
        # Catch exceptions raised by GoogleTranslator (e.g., unsupported language)
        return None


def detect_user_language(text):
    """Detects the language of the input text using langdetect."""
    if not text or text.strip() == "":
        return 'en'
    try:
        print(f"[Attempting language detection for: '{text}']")
        # Primary detection using langdetect
        lang = detect(text)
        print(f"[Detected User Language (langdetect): {lang}]")
        # Return the detected code. We will handle whether it's supported for speaking/translation elsewhere.
        return lang
    except LangDetectException:
        print("[Language Detection Failed (langdetect)] Defaulting to English.")
        return "en"
    except Exception as e:
        print(f"[Unexpected Language Detection Error: {e}] Defaulting to English.")
        return "en"


def match_intent(user_input):
    """Matches user input to the best intent using TF-IDF."""
    # Move global declarations to the top of the function
    global X, vectorizer, intent_phrases, intent_tags # Moved to the top

    if not user_input:
        return None, 0.0

    # Ensure the vectorizer is fitted before attempting to transform
    if 'X' not in globals() or X is None:
         print("[ERROR] TF-IDF vectorizer is not fitted. Cannot match intent.")
         # Attempt to re-fit if possible, though this indicates a setup issue
         try:
             print("[Attempting to re-fit vectorizer]")
             # global X, vectorizer, intent_phrases, intent_tags # Removed from here

             intent_phrases = []
             intent_tags = []
             for tag, phrases in intents.items():
                  if tag != "set_language_mode":
                       intent_phrases.extend(phrases)
                       intent_tags.extend([tag] * len(phrases))
                  else:
                       intent_phrases.extend(["change language", "switch language", "set language"])
                       intent_tags.extend([tag] * 3)
             vectorizer = TfidfVectorizer()
             X = vectorizer.fit_transform(intent_phrases)
             print("[INFO] TF-IDF vectorizer re-fitted successfully.")
             # Retry matching with the newly fitted vectorizer
             user_vec = vectorizer.transform([user_input])
             scores = cosine_similarity(user_vec, X)
             best_match_idx = scores.argmax()
             best_score = scores[0, best_match_idx]
             confidence_threshold = 0.4
             matched_tag = intent_tags[best_match_idx]
             if best_score >= confidence_threshold:
                  print(f"[Intent Matched (Re-fit): {matched_tag} with confidence {round(best_score, 2)}]")
                  return matched_tag, best_score
             else:
                  print(f"[Low Confidence Match (Re-fit): {matched_tag} ({round(best_score, 2)})]")
                  return "unknown", best_score

         except Exception as re_fit_e:
              print(f"[ERROR] Failed to re-fit TF-IDF vectorizer: {re_fit_e}")
              return "unknown", 0.0


    try:
        user_vec = vectorizer.transform([user_input])
        scores = cosine_similarity(user_vec, X)
        best_match_idx = scores.argmax()
        best_score = scores[0, best_match_idx]
        confidence_threshold = 0.4 # Adjusted threshold

        matched_tag = intent_tags[best_match_idx]

        # Higher confidence for critical intents
        if matched_tag == "emergency_call" and best_score < 0.55:
             print(f"[Intent Match: emergency_call, but confidence too low ({round(best_score, 2)})]")
             return "unknown", best_score # Treat as unknown if confidence is low

        # We are now handling language mode and specific greetings/translations with regex *before* intent matching,
        # so the TF-IDF match here is for more general or less specific phrases.

        if best_score >= confidence_threshold:
            print(f"[Intent Matched: {matched_tag} with confidence {round(best_score, 2)}]")
            return matched_tag, best_score
        else:
            print(f"[Low Confidence Match: {matched_tag} ({round(best_score, 2)})]")
            return "unknown", best_score
    except Exception as e:
        print(f"[Intent Matching Error]: {e}")
        return "unknown", 0.0

def extract_name(user_input):
    """
    Extracts name using a more precise regex for 'my name is' or 'i am'.
    Aims to capture the name immediately following the phrase.
    """
    global user_name
    # More precise regex: captures words immediately following "my name is" or "i am/i'm"
    # It stops capturing at punctuation, common non-name words, or the end of the string.
    # Added more non-name words to the negative lookahead.
    match = re.search(r"(?:my name is|i am|i'm)\s+([a-zA-Z\s]+?)(?:\.|!|\?|,|\s+(?:and|so|but|because|which|what|how|when|where|why|is|am|are|was|were|have|has|had|do|did|don't|can|can't|will|won't|would|should|could|if|then|than|or|nor|for|at|in|on|of|to|from|by|with|about|as|at|by|for|from|in|into|like|of|off|on|out|over|past|since|through|to|under|up|with|your|my|his|her|their|our)\b|$)", user_input.lower())

    extracted = None
    if match:
        extracted_potential = match.group(1).strip()
        # Further refine by splitting and taking the first few words, and basic validation
        words = extracted_potential.split()
        if words:
            # Assuming names are typically 1 to 4 words (to handle names like "Mary Ann")
            potential_name = " ".join(words[:4]).title()
            # Basic validation: check length and avoid common non-names or single letters
            if len(potential_name) > 1 and potential_name.lower() not in ["is", "am", "me", "joey", "in", "to", "a", "the", "i", "you", "what", "how", "when", "where", "why", "and", "so", "but", "for", "my", "name", "good", "good spanish", "good hindi"]: # Added "good spanish", "good hindi" to exclusion
                 extracted = potential_name

    if extracted:
        user_name = extracted
        print(f"User name set to: {user_name}")
        return user_name
    else:
         print("[INFO] Could not extract a valid name.")
         return None


# --- Placeholder Functions (Keep as before) ---
def get_location():
    print("[Placeholder] Returning mock location.")
    return "New Delhi, Delhi, India"
def get_weather():
    print("[Placeholder] Returning mock weather.")
    # Simulate fetching weather data - in a real app, use a weather API
    try:
        # Example using a hypothetical weather API call
        # weather_data = weather_api.get_weather("your_location")
        # return weather_data
        # Using mock data for now:
        return {"location": "New Delhi", "temp_c": 35, "condition": "mostly sunny", "temp_f": round(35 * 9/5 + 32)}
    except Exception as e:
        print(f"[Placeholder Weather Error]: {e}")
        return None # Indicate failure

def get_current_speed():
    print("[Placeholder] Returning mock speed.")
    # Simulate reading speed from a sensor or system - in a real app, integrate with car's system
    return random.randint(30, 80) # km/h

def get_traffic_signal_status():
    print("[Placeholder] Returning mock traffic signal.")
    # Simulate reading traffic signal status - in a real app, use traffic data API or camera
    return random.choices(['green', 'yellow', 'red'], weights=[10, 1, 2])[0] # More likely to be green

def simulate_heartbeat():
    print("[Placeholder] Returning mock heartbeat.")
    # Simulate reading from a biometric sensor
    return random.randint(60, 100) # beats per minute

# Input/Output and Feature Handlers
def listen():
    """Listens for user input via microphone."""
    with sr.Microphone() as source:
        print("\nAdjusting for ambient noise...")
        try:
            # Adjust for ambient noise dynamically
            recognizer.adjust_for_ambient_noise(source, duration=1.5)
            print("Listening...")
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=15)
            print("Processing...")
            # Use Google's speech recognition
            text = recognizer.recognize_google(audio)
            print(f"You: {text}")
            return text.lower() # Return lowercased text for easier matching
        except sr.WaitTimeoutError:
            # print("Listening timed out while waiting for phrase to start")
            return "" # Return empty string on timeout
        except sr.UnknownValueError:
            # API was unable to understand the speech
            print("Didn't catch that.")
            return ""
        except sr.RequestError as e:
            # API was unreachable or unresponsive
            print(f"Service unavailable; {e}")
            speak("Sorry, I'm having trouble connecting to the speech service.", 'en')
            return ""
        except Exception as e:
            print(f"Error in listen(): {e}")
            return "" # Return empty string on other errors


def handle_emergency(response_lang):
    """Handles the emergency call action."""
    responses = {
        'en': "Emergency situation detected. Calling emergency services now. Please remain calm.",
        'hi': "आपातकालीन स्थिति का पता चला। आपातकालीन सेवाओं को कॉल किया जा रहा है। कृपया शांत रहें।",
        'es': "Situación de emergencia detectada. Llamando a los servicios de emergencia ahora. Por favor, mantén la calma.",
        'ur': "ہنگامی صورتحال کا پتہ چلا۔ ایمرجنسی سروسز کو کال کی جا رہی ہے۔ براہ کرم پرسکون رہیں۔",
        'bn': "জরুরী অবস্থা সনাক্ত করা হয়েছে। জরুরী পরিষেবাগুলিতে কল করা হচ্ছে। শান্ত থাকুন।" # Added Bengali
    }
    speak(responses.get(response_lang, responses['en']), response_lang)
    print(">>> SIMULATING CALL TO EMERGENCY NUMBER (e.g., 112)... <<<")
    # TODO: Implement actual emergency contact/service integration here.


def handle_distress_signal(user_input, user_lang):
    """Checks for distress signals and initiates emergency protocol if needed."""
    # Determine if the user input strongly indicates an emergency, potentially overriding intent matching
    critical_emergency_patterns = [
        r'\bi need help\b', r'\bsos\b', r'\bfire emergency\b', r'\bmedical emergency\b',
        r'\bmujhe madad chahiye\b', # Hindi
        r'\bnecesito ayuda\b', r'\bemergencia médica\b', r'\bemergencia de incendio\b', # Spanish
        r'\bmujhay madad chahiye\b', r'\bمیڈیکل ایمرجنسی\b', r'\bآگ لگی ہے\b', # Urdu
        r'\bআমার সাহায্য দরকার\b', r'\bমেডিকেল ইমার্জেন্সি\b', r'\bফায়ার ইমার্জেন্সি\b' # Bengali examples - need to add these to intents too
    ]

    is_distress = False
    for pattern in critical_emergency_patterns:
        if re.search(pattern, user_input, re.IGNORECASE):
            print(f"[Critical Emergency Pattern Matched: {pattern}]")
            is_distress = True
            break # Found a critical pattern, no need to check others

    # Also consider the intent match if confidence is high enough for emergency_call
    # Pass user_input to match_intent
    intent, score = match_intent(user_input) # Re-calculate intent here for the most current input check
    if intent == "emergency_call" and score > 0.7: # Higher confidence for intent-based trigger
        print(f"[Emergency Call Intent Matched with high confidence: {score}]")
        is_distress = True

    if is_distress:
        responses_check = {
            'en': "It sounds like you might be in distress. Initiating emergency procedures now.",
            'hi': "लगता है आप संकट میں ہیں। اب आपातकालीन प्रक्रिया شروع کر رہا ہوں", # Corrected Hindi/Urdu mix
            'es': "Parece que podrías estar en peligro. Iniciando procedimientos de emergencia ahora.",
            'ur': "ایسا لگتا ہے کہ آپ پریشانی میں ہیں۔ اب ہنگامی طریقہ کار شروع کر رہا ہوں۔",
            'bn': "মনে হচ্ছে আপনি সংকটে আছেন। জরুরি পদ্ধতি এখন শুরু করা হচ্ছে।" # Added Bengali
        }
        speak(responses_check.get(user_lang, responses_check['en']), user_lang)
        handle_emergency(user_lang)
        return True # Indicate distress was handled
    return False # Indicate no distress signal handled


# === Main Interaction Loop ===
def main():
    global user_name, active_language_mode

    # Initial greeting - ask for name if not known
    initial_greeting_lang = active_language_mode if active_language_mode else 'en'
    if user_name:
         speak(f"Hello {user_name}, Joey is ready.", initial_greeting_lang)
    else:
         speak("Hello there, I am Joey. What's your name?", initial_greeting_lang)


    while True:
        # --- Background Tasks (Optional - needs threading for non-blocking operation) ---
        # If these checks take significant time, they will block listening.
        # For a real-time assistant, you'd run these in separate threads or processes.
        # check_and_warn_speeding()
        # check_and_warn_traffic_light()

        # --- Listen ---
        user_input = listen()
        if not user_input:
            time.sleep(0.5) # Small delay if no input
            continue

        # --- Determine Response Language ---
        # Prioritize active language mode. If no active mode, detect input language for potential future use
        detected_input_lang = detect_user_language(user_input) # Detect input language
        response_lang = active_language_mode if active_language_mode else 'en' # Response language is active mode or default English

        print(f"[Current Response Language: {response_lang}] (Detected Input Language: {detected_input_lang})")


        # --- Check for Language Mode Toggles (Priority Handling using regex) ---
        # These should be handled before intent matching and should explicitly change active_language_mode
        mode_changed = False
        temp_response_text = ""
        temp_response_lang_confirm = 'en' # Default language for confirming mode change

        # Regex to capture potential language name and on/off/mode state
        # This regex specifically looks for a language name followed by mode/on/off terms or 'in/to'.
        # Use word boundaries (\b) to match whole words.
        # Added 'speak in' as a trigger for mode change as well.
        mode_match = re.search(r"\b(?:speak\s+in\s+)?(?:" + "|".join(re.escape(name) for name in LANGUAGE_CODES.keys()) + r")\s*(?:mode\s+|modo\s+|मोड\s+)?(?:on|off|चालू|बंद|activar|desactivar|آن|آف|शुरू)?\b", user_input, re.IGNORECASE) # Added speak in

        if mode_match:
            mode_changed = True
            lang_part = mode_match.group(0).lower() # Get the matched language and mode part

            # Find the language code from the matched part
            requested_lang_code = None
            for lang_name, code in LANGUAGE_CODES.items():
                 # Use a word boundary to match whole language names within the matched part
                 if re.search(r'\b' + re.escape(lang_name) + r'\b', lang_part):
                      requested_lang_code = code
                      break

            if requested_lang_code:
                 # Check if the phrase implies "on" or "off" or just setting the language
                 is_on = re.search(r"(on|चालू|शुरू|activar|آن)\b", lang_part) is not None
                 is_off = re.search(r"(off|बंद|desactivar|آف)\b", lang_part) is not None
                 # Check for phrases like "speak in [language]" or "[language] mode" without explicit on/off
                 is_set = re.search(r"\bspeak\s+in\s+\b" + re.escape(lang_name) + r"\b", lang_part, re.IGNORECASE) is not None or re.search(r"\b" + re.escape(lang_name) + r"\b(?: mode)?\b", lang_part, re.IGNORECASE) is not None


                 if is_on or is_set:
                      if active_language_mode != requested_lang_code:
                           active_language_mode = requested_lang_code
                           # Attempt to speak confirmation in the requested language
                           temp_response_text_en = f"Okay, switching to {requested_lang_code} mode."
                           # Translate confirmation message if possible, otherwise use English
                           # Ensure translation uses the correct code and check for None
                           translated_confirm = translate_text(temp_response_text_en, requested_lang_code)
                           temp_response_text = translated_confirm if translated_confirm else temp_response_text_en
                           temp_response_lang_confirm = requested_lang_code # Try to confirm in the new language
                      else:
                           temp_response_text_en = f"I am already in {requested_lang_code} mode."
                           translated_confirm = translate_text(temp_response_text_en, requested_lang_code)
                           temp_response_text = translated_confirm if translated_confirm else temp_response_text_en
                           temp_response_lang_confirm = requested_lang_code # Try to confirm in the active language
                 elif is_off:
                      if active_language_mode == requested_lang_code:
                           active_language_mode = None # Setting to None means default (English)
                           temp_response_text = f"Okay, {requested_lang_code} mode turned off. Switching to default English."
                           temp_response_lang_confirm = 'en'
                      elif active_language_mode is None and requested_lang_code == 'en':
                            temp_response_text = "I am already in default English mode."
                            temp_response_lang_confirm = 'en'
                      else:
                           # If they say "Spanish off" but aren't in Spanish mode
                           temp_response_text = f"Okay, turning off {requested_lang_code} mode (if it was on). Switching to default English."
                           temp_response_lang_confirm = 'en'
            else:
                # This case should be less likely with the regex, but good to have a fallback
                temp_response_text = "Sorry, I didn't recognize that language mode request."
                temp_response_lang_confirm = 'en'


        if mode_changed:
            speak(temp_response_text, temp_response_lang_confirm)
            # After changing mode, the response_lang for the *next* turn will reflect the change
            continue # Skip subsequent processing for this turn


        # --- Handle Distress Signals (High Priority) ---
        # Check for distress signals regardless of language mode
        if handle_distress_signal(user_input, response_lang): # Use response_lang for speaking the confirmation
             continue # If distress is handled, skip normal intent processing


        # --- Handle Specific Fixed Phrases (Highest Priority) ---
        # Handle the "say hello to our boss" request - must come BEFORE generic greet_someone
        # Regex to capture the optional language at the end
        boss_match = re.search(r"say hello to our boss\s*(?:in\s+([a-zA-Z]+))?", user_input, re.IGNORECASE)
        if boss_match:
             target_lang_name = boss_match.group(1) # Capture the language name if present
             boss_greeting_en = "Hello Tushkit Gupta!" # The English phrase to translate

             if target_lang_name:
                  target_lang_code = get_language_code(target_lang_name)
                  if target_lang_code:
                       translated_greeting = translate_text(boss_greeting_en, target_lang_code)
                       if translated_greeting:
                            speak(translated_greeting, target_lang_code) # Speak the translated greeting in the target language
                       else:
                            # Fallback if translation fails
                            speak(f"Sorry, I couldn't translate that greeting to the requested language. Saying it in English.", response_lang) # Speak error in current response lang
                            speak(boss_greeting_en, response_lang) # Speak English greeting in current language
                  else:
                       speak(f"Sorry, I don't recognize the language '{target_lang_name}' for this greeting. Saying hello to the boss in English.", response_lang) # Speak error in current response lang
                       speak(boss_greeting_en, response_lang) # Speak English greeting in current language

             else:
                  # If no language specified, speak the English greeting in the current response_lang
                  speak(boss_greeting_en, response_lang)

             continue # Skip normal intent processing for this specific command


        # --- Handle Specific Greeting with Name and Optional Language (Issue 1 Fix) ---
        # Handle "say hello to [name]" or "say hello to [name] in [language]"
        # Regex to capture the name after the greeting phrase, optionally followed by 'in [language]'
        # Group 1: the name part (non-greedy)
        # Group 2: the language name after 'in ' or 'to '
        greet_name_match = re.search(r"(?:say hello to|say hi to|greet|give my regards to|tell)\s+([a-zA-Z\s]+?)(?:\s+(?:in|to)\s+([a-zA-Z]+))?$", user_input, re.IGNORECASE)

        if greet_name_match:
             person_name_part = greet_name_match.group(1).strip()
             target_lang_name_part = greet_name_match.group(2) # Captured language name if present

             person_name = None
             target_lang_code = None

             # Basic processing for the name part
             words = person_name_part.split()
             if words:
                 # Take up to the first 4 words, capitalize
                 potential_name = " ".join(words[:4]).title()
                 # Basic validation for the name (avoiding single letters or common short words)
                 if len(potential_name) > 1 and potential_name.lower() not in ["a", "the", "i", "you", "me", "him", "her", "us", "them", "joey", "boss", "our boss", "someone"]: # Added "someone"
                      person_name = potential_name

             if target_lang_name_part:
                  target_lang_code = get_language_code(target_lang_name_part)
                  if not target_lang_code:
                       print(f"[Greeting Extraction] Unrecognized language specified: '{target_lang_name_part}'")


             if person_name: # If a valid name was extracted
                  # Construct the basic greeting phrase in English
                  base_greeting_en = f"Hello {person_name}!"

                  # Translate the greeting if a valid target language was specified
                  if target_lang_code:
                       translated_greeting = translate_text(base_greeting_en, target_lang_code)
                       if translated_greeting:
                            speak(translated_greeting, target_lang_code) # Speak the translated greeting in the target language
                       else:
                            # Fallback if translation fails
                            speak(f"Sorry, I couldn't translate 'Hello {person_name}!' to {target_lang_code}. Saying it in English.", response_lang) # Speak error in current response lang
                            speak(base_greeting_en, response_lang) # Speak English greeting in current language
                  else:
                       # If no specific language was requested, speak the English greeting in the current response_lang
                       speak(base_greeting_en, response_lang) # Use response_lang

             elif person_name_part.lower() == "our boss":
                  # This specific case is handled by the high-priority check at the start of the loop
                  pass # Do nothing here, it was handled by the boss_match regex check

             elif person_name_part.lower() == "someone":
                   # This is a generic greet someone request, let the TF-IDF intent handle it
                   # Do nothing here, let the intent matching proceed
                   print("[INFO] Generic 'greet someone' matched regex, proceeding to TF-IDF.")
                   pass # Continue to intent matching

             else:
                   # If the regex matched the pattern but couldn't extract a valid name
                   responses = {
                         'en': "Sorry, I didn't catch the name of the person you want me to greet.",
                         'hi': "माफ़ करना, मुझे उस व्यक्ति का नाम समझ नहीं आया जिसे आप नमस्ते कहना चाहते हैं।",
                         'es': "Lo siento, no entendí el nombre de la persona que quieres que salude.",
                         'ur': "معاف کرنا، مجھے اس شخص کا نام سمجھ نہیں آیا جسے آپ سلام کہنا چاہتے ہیں۔",
                         'bn': "দুঃখিত, আপনি কাকে হ্যালো বলতে চান তা বুঝতে পারিনি।" # Added Bengali
                    }
                   speak(responses.get(response_lang, responses['en']), response_lang)
             # We handle specific greetings here, so we continue to the next iteration if one was matched
             continue


        # --- Handle Translate Request (Issue 2 Fix) ---
        # Handle phrases like "translate X to Y" or "say X in Y"
        # This regex captures the text to translate (Group 1) and the target language (Group 2)
        # Revised regex to capture the text more reliably before "in/to [language]".
        # It looks for the intro phrase, then captures everything (greedy .*)
        # until it finds " in " or " to " followed by letters.
        translate_match = re.search(r"^(?:translate|say|how do you say|tell me to say)\s+(.*)\s+(?:in|to)\s+([a-zA-Z]+)$", user_input, re.IGNORECASE)

        if translate_match:
             # Group 1 is the text to translate, Group 2 is the language name
             text_to_translate = translate_match.group(1).strip()
             target_language_name = translate_match.group(2).strip().lower()

             target_lang_code = get_language_code(target_language_name)

             if text_to_translate and target_lang_code:
                  translated_text = translate_text(text_to_translate, target_lang_code)
                  if translated_text:
                       # Speak the translated text in the target language
                       speak(translated_text, target_lang_code)
                  else:
                       # Fallback if translation fails (e.g., unsupported language by translator)
                       speak(f"Sorry, I couldn't translate '{text_to_translate}' to {target_language_name}.", response_lang) # Speak error in current response lang
             elif target_language_name and not target_lang_code:
                  # If target language is not recognized
                  speak(f"Sorry, I don't recognize the language '{target_language_name}' for translation.", response_lang) # Speak error in current response lang
             else:
                  # This case should be less likely with the new regex, but include fallback
                  responses = {
                  'en': "What would you like me to translate and to which language?",
                  'hi': "आप क्या अनुवाद करना चाहेंगे और किस भाषा में?",
                  'es': "¿Qué te gustaría que tradujera y a qué idioma?",
                  'ur': "آپ کیا ترجمہ کرنا چاہیں گے اور کس زبان میں؟",
                  'bn': "আপনি কি অনুবাদ করতে চান এবং কোন ভাষায়?" # Added Bengali
                  }
                  speak(responses.get(response_lang, responses['en']), response_lang)

             # Continue to the next iteration after handling translation
             continue


        # --- Intent Matching ---
        # Perform TF-IDF matching only if the input wasn't handled by high-priority regex checks
        # Note: The original 'translate' intent will still be matched by TF-IDF for general phrases
        # like "translate this", but the more specific regex above will handle "translate X to Y".
        intent, score = match_intent(user_input)


        # --- Intent Handling ---
        # The response language for these intents will be based on active_language_mode (response_lang)
        if intent == "greet":
            greetings = {
                'en': ["Hello!", "Hi there!", "Hey!", "Greetings!", "Good to hear from you!"],
                'hi': ["नमस्ते!", "हाय!", "हैलो!", "आपसे सुनकर अच्छा लगा!"],
                'es': ["¡Hola!", "¡Qué tal!", "¡Saludos!", "¡Me alegra escucharte!"],
                'ur': ["اسلام علیکم!", "آداب!", "سلام!", "آپ سے سن کر اچھا لگا!"],
                'bn': ["হ্যালো!", "নমস্কার!", "কেমন আছেন?", "শুনে ভালো লাগলো!"], # Added Bengali greetings
                'ja': ["こんにちは！"], # Added Japanese greeting
                'de': ["Hallo!"], # Added German greeting
                # Add greetings for other supported languages
            }
            # Use the user's name if known
            # Use get_language_code for robust lookup in greetings dictionary
            greeting_text = random.choice(greetings.get(get_language_code(response_lang), greetings['en'])) # Fallback to English if response_lang not in greetings
            if user_name:
                 greeting_text += f" {user_name}"
            speak(greeting_text, response_lang)

        elif intent == "greet_someone":
             # This branch is for generic "greet someone" if the specific "say hello to [name]..." regex didn't match
             # It won't handle specific names or languages as that was done by regex.
             responses = {
                  'en': "Okay, I can greet someone if you tell me their name.",
                  'hi': "ठीक है, अगर आप मुझे उनका नाम बताएं तो मैं किसी का अभिवादन कर सकता हूँ।",
                  'es': "De acuerdo, puedo saludar a alguien si me dices su nombre.",
                  'ur': "ٹھیک ہے، اگر آپ مجھے ان کا نام بتائیں تو میں کسی کو سلام کر سکتا ہوں۔",
                  'bn': "ঠিক আছে، আপনি যদি আমাকে তাদের নাম বলেন তবে আমি কাউকে অভিবাদন জানাতে পারি।" # Added Bengali
             }
             speak(responses.get(response_lang, responses['en']), response_lang)


        elif intent == "ask_for_help":
            responses = {
                'en': "I can tell you the time, weather, tell jokes, translate to many languages, remember your name, and more. Just ask!",
                'hi': "मैं आपको समय, मौसम बता सकता हूँ, चुटकुले सुना सकता हूँ, कई भाषाओं में अनुवाद कर सकता हूँ، आपका नाम याद रख सकता ہوں، اور بھی بہت کچھ۔ بس پوچھیں!", # Corrected Hindi/Urdu mix
                'es': "Puedo decirte la hora, el clima, contar chistes, traducir a muchos idiomas, recordar tu nombre y más. ¡Solo pregunta!",
                'ur': "میں آپ کو وقت، موسم بتا سکتا ہوں، لطیفے سنا سکتا ہوں، کئی زبانوں میں ترجمہ کر سکتا ہوں، آپ کا نام یاد رکھ سکتا ہوں، اور بہت کچھ۔ بس پوچھیں!",
                'bn': "আমি আপনাকে সময়, আবহাওয়া বলতে পারি, কৌতুক বলতে পারি, অনেক ভাষায় অনুবাদ করতে পারি، আপনার নাম মনে রাখতে পারি এবং আরও অনেক কিছু করতে পারি। শুধু জিজ্ঞাসা করুন!" # Added Bengali
            }
            speak(responses.get(response_lang, responses['en']), response_lang)

        # Emergency call is handled by handle_distress_signal for higher priority check
        # elif intent == "emergency_call":
        #     handle_emergency(response_lang)

        elif intent == "tell_a_joke":
            # Use get_language_code for robust lookup in jokes_multi dictionary
            jokes = jokes_multi.get(get_language_code(response_lang), jokes_multi['en']) # Fallback to English jokes
            if jokes:
                speak(random.choice(jokes), response_lang)
            else:
                 responses = {
                      'en': "Sorry, I don't have any jokes in that language right now.",
                      'hi': "माफ़ करना, मेरे पास अभी उस भाषा में کوئی चुٹکلے نہیں ہیں۔", # Corrected Hindi/Urdu mix
                      'es': "Lo siento, no tengo chistes en ese idioma en este momento.",
                      'ur': "معاف کرنا، میرے پاس فی الحال اس زبان میں کوئی لطیفے نہیں ہیں۔",
                      'bn': "দুঃখিত، আমার কাছে এই মুহূর্তে ঐ ভাষায় কোনো কৌতুক নেই।" # Added Bengali
                 }
                 speak(responses.get(response_lang, responses['en']), response_lang)


        elif intent == "joke_feedback_negative":
             responses = {
                  'en': "Oh, I'm sorry you didn't find that funny. I'll try to find better jokes for you!",
                  'hi': "ماف کرنا، مجھے ماف کرنا اگر آپ کو وہ مضحکہ خیز نہیں لگا۔ میں آپ کے لیے بہتر لطیفے ڈھونڈنے کی کوشش کروں گا۔", # Corrected Hindi/Urdu mix
                  'es': "Oh, lamento que no te haya parecido divertido. ¡Intentaré encontrar mejores chistes para ti!",
                  'ur': "اوہ، مجھے افسوس ہے کہ آپ کو یہ مضحکہ خیز نہیں لگا۔ میں آپ کے لیے بہتر لطیفے تلاش کرنے کی کوشش کروں گا!",
                  'bn': "ওহ, আমি দুঃখিত আপনি এটা মজার খুঁজে পাননি। আমি আপনার জন্য আরও ভালো কৌতুক খুঁজে বের করার চেষ্টা করব!" # Added Bengali
             }
             speak(responses.get(response_lang, responses['en']), response_lang)


        elif intent == "thank_you":
            responses = {
                'en': ["You're welcome!", "No problem!", "Anytime!", "Glad I could help!"],
                'hi': ["आपका स्वागत है!", "कोई बात नहीं!", "कभी भी!", "खुशी हुई कि मैं मदद कर सका!"],
                'es': ["¡De nada!", "¡No hay problema!", "¡Cuando quieras!", "¡Me alegra haber podido ayudar!"],
                'ur': ["خوش آمدید!", "کوئی بات نہیں!", "جب چاہیں!", "خوشی ہوئی کہ میں مدد کر سکا!"],
                'bn': ["আপনাকে স্বাগতম!", "কোন সমস্যা নেই!", "যেকোনো সময়!", "সাহায্য করতে পেরে ভালো লাগছে!"] # Added Bengali
            }
            speak(random.choice(responses.get(response_lang, responses['en'])), response_lang)

        elif intent == "stop_or_exit":
            responses = {
                'en': "Goodbye! Have a great day!",
                'hi': "अलविदा! आपका दिन शानदार हो!",
                'es': "¡Adiós! ¡Que tengas un gran día!",
                'ur': "اللہ حافظ! آپ کا دن اچھا گزرے!",
                'bn': "বিদায়! আপনার দিনটি দারুণ কাটুক!" # Added Bengali
            }
            speak(responses.get(response_lang, responses['en']), response_lang)
            break # Exit the loop

        elif intent == "introduce_myself":
            # This intent is triggered by phrases like "my name is", "i am", etc.
            # Extract the name and set the user_name global variable
            extracted = extract_name(user_input) # Use the improved extract_name

            if extracted:
                responses = {
                    'en': f"Nice to meet you, {extracted}! I'll remember your name.",
                    'hi': f"آپ سے مل کر اچھا لگا، {extracted}! میں آپ کا نام یاد رکھوں گا۔", # Corrected Hindi/Urdu mix
                    'es': f"Encantado de conocerte, {extracted}! Recordaré tu nombre.",
                    'ur': f"آپ سے مل کر اچھا لگا، {extracted}! میں آپ کا نام یاد رکھوں گا۔",
                    'bn': f"আপনার সাথে দেখা করে ভালো লাগলো، {extracted}! আমি আপনার নাম মনে রাখব।" # Added Bengali
                }
                speak(responses.get(response_lang, responses['en']), response_lang)

                # --- Handle the "what's yours" part if present after introduction ---
                if re.search(r"(what'?s yours|and your name|aur tumhara naam)", user_input, re.IGNORECASE):
                     responses_name = {
                          'en': "My name is Joey.",
                          'hi': "میرا نام جॉय ہے۔", # Corrected Hindi/Urdu mix
                          'es': "Mi nombre es Joey.",
                          'ur': "میرا نام جَوی ہے۔",
                          'bn': "আমার নাম জয়ে।" # Added Bengali
                     }
                     speak(responses_name.get(response_lang, responses_name['en']), response_lang)

            else:
                # If name extraction failed for the introduce_myself intent
                responses = {
                    'en': "Sorry, I couldn't catch your name. Could you please repeat it?",
                    'hi': "ماف کرنا، میں آپ کا نام سمجھ نہیں پایا۔ کیا آپ کر پیا اسے دوبارہ کہہ سکتے ہیں؟", # Corrected Hindi/Urdu mix
                    'es': "Lo siento, no pude entender tu nombre. ¿Podrías repetirlo por favor?",
                    'ur': "معاف کرنا، میں آپ کا نام سمجھ نہیں پایا۔ کیا آپ براہ کرم اسے دہرا سکتے ہیں؟",
                    'bn': "দুঃখিত، আমি আপনার নাম বুঝতে পারিনি। আপনি কি দয়া করে এটি পুনরাবৃত্তি করতে পারেন?" # Added Bengali
                }
                speak(responses.get(response_lang, responses['en']), response_lang)

        elif intent == "ask_name":
             if user_name:
                  responses = {
                       'en': f"Your name is {user_name}.",
                       'hi': f"آپ کا نام {user_name} ہے۔", # Corrected Hindi/Urdu mix
                       'es': f"Tu nombre es {user_name}.",
                       'ur': f"آپ کا نام {user_name} ہے۔",
                       'bn': f"আপনার نام {user_name}।" # Added Bengali
                  }
                  speak(responses.get(response_lang, responses['en']), response_lang)
             else:
                  responses = {
                       'en': "I don't know your name yet. You can tell me by saying, 'My name is [your name]'.",
                       'hi': "مجھے ابھی آپ کا نام نہیں پتا۔ آپ مجھے 'میرا نام [آپ کا نام] ہے' کہہ کر بتا سکتے ہیں۔", # Corrected Hindi/Urdu mix
                       'es': "Aún no sé tu nombre. Puedes decírmelo diciendo: 'Mi nombre es [tu nombre]'.",
                       'ur': "مجھے ابھی آپ کا نام نہیں پتا۔ آپ مجھے 'میرا نام [آپ کا نام] ہے' کہہ کر بتا سکتے ہیں۔",
                       'bn': "আমি এখনও আপনার নাম জানি না। আপনি আমাকে 'আমার নাম [আপনার নাম]' বলে বলতে পারেন।" # Added Bengali
                  }
                  speak(responses.get(response_lang, responses['en']), response_lang)

        elif intent == "about_joey":
             # Add specific checks for questions about attributes and provide more detailed responses
             spoken_a_specific_response = False

             if re.search(r"hair color|hair colour|baal|pelo|بال|do you not have hair|you don't have hair|kya tumhare baal nahin hain|¿no tienes pelo?|kya aap ke baal nahi hain", user_input, re.IGNORECASE):
                  responses_hair = {
                       'en': "As an AI, I don't have a physical body like humans do, so I don't have hair or a hair color. I exist as computer code and data.",
                       'hi': "ایک AI ہونے کے ناطے، میرا انسانوں جیسا کوئی भौतिक शरीर نہیں ہے، اسی لیے میرے بال یا بالوں کا رنگ نہیں ہے۔ میں کمپیوٹر کوڈ اور ڈیٹا کے طور پر موجود ہوں۔", # Corrected Hindi/Urdu mix
                       'es': "Como IA, no tengo un cuerpo físico como los humanos, así que no tengo pelo ni color de pelo. Existo como código y datos de computadora.",
                       'ur': "ایک AI کے طور پر، میرا انسانوں جیسا کوئی جسمانی جسم نہیں ہے، لہذا میرے بال یا بالوں کا رنگ نہیں ہے۔ میں کمپیوٹر کوڈ اور ڈیٹا کے طور پر موجود ہوں۔",
                       'bn': "একজন এআই হিসেবে، আমার মানুষের মতো শারীরিক শরীর নেই، তাই আমার চুল বা চুলের রঙ নেই। আমি কম্পিউটার কোড এবং ডেটা হিসেবে বিদ্যমান।" # Added Bengali
                  }
                  speak(responses_hair.get(response_lang, responses_hair['en']), response_lang)
                  spoken_a_specific_response = True

             elif re.search(r"age|umar|edad|عمر", user_input, re.IGNORECASE):
                  responses_age = {
                       'en': "I don't have a traditional age in the human sense. My development is ongoing, but I was last updated on [Insert Date/Version Info if available].", # You could make this more specific
                       'hi': "میری انسانوں والی کوئی روایتی عمر نہیں ہے۔ میری ترقی جاری ہے، لیکن مجھے آخری بار [اگر دستیاب ہو تو تاریخ/ورژن کی معلومات ڈالیں] کو اپ ڈیٹ کیا گیا تھا۔", # Corrected Hindi/Urdu mix
                       'es': "No tengo una edad tradicional en el sentido humano. Mi desarrollo es continuo, but I was last updated on [Insert Date/Version Info if available].", # Corrected Spanish
                       'ur': "میری انسانی معنوں میں کوئی روایتی عمر نہیں ہے۔ میری ترقی جاری ہے، لیکن مجھے آخری بار [اگر دستیاب ہو تو تاریخ/ورژن کی معلومات داخل کریں] کو اپ ڈیٹ کیا گیا تھا۔",
                       'bn': "মানুষের অর্থে আমার কোনো প্রচলিত বয়স নেই। আমার উন্নয়ন চলমান، তবে আমাকে শেষবার [যদি উপলব্ধ থাকে তবে তারিখ/সংস্করণ তথ্য ঢোকান] তারিখে আপডেট করা হয়েছিল।" # Added Bengali
                  }
                  speak(responses_age.get(response_lang, responses_age['en']), response_lang)
                  spoken_a_specific_response = True

             elif re.search(r"who made you|who created you", user_input, re.IGNORECASE):
                  responses_creator = {
                       'en': "I am a large language model, trained by Google.",
                       'hi': "میں گوگل کی طرف سے تربیت یافتہ ایک بڑا زبانی ماڈل ہوں", # Corrected Hindi/Urdu mix
                       'es': "Soy un modelo de lenguaje grande, entrenado por Google.",
                       'ur': "میں گوگل کے ذریعہ تربیت یافتہ ایک بڑا لسانی ماڈل ہوں۔",
                       'bn': "আমি গুগল দ্বারা প্রশিক্ষিত একটি বৃহৎ ভাষা মডেল।" # Added Bengali
                  }
                  speak(responses_creator.get(response_lang, responses_creator['en']), response_lang)
                  spoken_a_specific_response = True

             elif re.search(r"what languages can you speak|speak any language|what languages do you know|kaun kaun si bhasha bol sakte ho|qué idiomas puedes hablar|kaun kaun si zaban bol saktay hain|koi bhi zaban bolen", user_input, re.IGNORECASE):
                  # Generate a list of supported languages from LANGUAGE_CODES
                  supported_langs_names = [name.title() for name in LANGUAGE_CODES.keys() if len(name) > 2 and name not in ['default', 'normal']] # Use names, filter short codes and modes
                  random.shuffle(supported_langs_names) # Shuffle to make it sound less robotic
                  # Format the list nicely (e.g., English, Hindi, Spanish, and many more.)
                  if len(supported_langs_names) > 7:
                       lang_list_text = ", ".join(supported_langs_names[:7]) + ", and many more."
                  else:
                       lang_list_text = ", ".join(supported_langs_names)

                  responses_languages = {
                       'en': f"I can communicate in several languages, including {lang_list_text}. My ability to speak depends on the available text-to-speech engines and translation services. You can ask me to switch modes or translate.",
                       'hi': f"میں کئی زبانوں میں بات چیت کر سکتا ہوں، جن میں شامل ہیں {lang_list_text}۔ میری بولنے کی صلاحیت دستیاب ٹیکسٹ ٹو سپیچ انجنوں اور ترجمہ کی خدمات پر منحصر ہے۔ آپ مجھے موڈ تبدیل کرنے یا ترجمہ کرنے کے لیے کہہ سکتے ہیں۔", # Corrected Hindi/Urdu mix
                       'es': f"Puedo comunicarme en varios idiomas, incluyendo {lang_list_text}. My ability to speak depends on the available text-to-speech engines and translation services. You can ask me to switch modes or translate.", # Corrected Spanish
                       'ur': f"میں کئی زبانوں میں بات چیت کر سکتا ہوں، جن میں شامل ہیں {lang_list_text}۔ میری بولنے کی صلاحیت دستیاب ٹیکسٹ ٹو سپیچ انجنوں اور ترجمہ کی خدمات پر منحصر ہے۔ آپ مجھے موڈ تبدیل کرنے یا ترجمہ کرنے کے لیے کہہ سکتے ہیں۔",
                       'bn': f"আমি বেশ কয়েকটি ভাষায় যোগাযোগ করতে পারি, যার মধ্যে রয়েছে {lang_list_text}। আমার কথা বলার ক্ষমতা উপলব্ধ টেক্সট-টু-স্পীচ ইঞ্জিন এবং অনুবাদ পরিষেবাগুলির উপর নির্ভর করে। আপনি আমাকে মোড পরিবর্তন করতে বা অনুবাদ করতে বলতে পারেন।" # Added Bengali
                  }
                  speak(responses_languages.get(response_lang, responses_languages['en']), response_lang)
                  spoken_a_specific_response = True

             elif re.search(r"are you real|are you alive|do you have feelings|are you a robot|are you human", user_input, re.IGNORECASE):
                   responses_nature = {
                        'en': "I am a computer program, an AI. I don't have feelings or a physical body, but I'm here to assist you.",
                        'hi': "میں ایک کمپیوٹر پروگرام ہوں، ایک AI۔ میرے احساس یا جسمانی جسم نہیں ہے، لیکن میں آپ کی مدد کے لیے یہاں ہوں۔", # Corrected Hindi/Urdu mix
                        'es': "Soy un programa de computadora, una IA. No tengo sentimientos ni cuerpo físico, but I'm here to assist you.", # Corrected Spanish
                        'ur': "میں ایک کمپیوٹر پروگرام ہوں، ایک AI۔ میرے احساسات یا جسمانی جسم نہیں ہے، لیکن میں آپ کی مدد کے لیے یہاں ہوں۔",
                        'bn': "আমি একটি কম্পিউটার প্রোগ্রাম، একটি এআই। আমার অনুভূতি বা শারীরিক শরীর নেই، তবে আমি আপনাকে সাহায্য করার জন্য এখানে আছি।" # Added Bengali
                   }
                   speak(responses_nature.get(response_lang, responses_nature['en']), response_lang)
                   spoken_a_specific_response = True


             # If no specific question about attributes is matched, give the general response
             if not spoken_a_specific_response:
                  responses_general_about = {
                     'en': "I am Joey, a voice assistant program designed to help you with various tasks.",
                     'hi': "میں جॉय ہوں، ایک وائس اسسٹنٹ پروگرام جسے آپ کی مختلف کاموں میں مدد کرنے کے لیے ڈیزائن کیا گیا ہے۔", # Corrected Hindi/Urdu mix
                     'es': "Soy Joey, un programa de asistente de voz diseñado para ayudarte con diversas tareas.",
                     'ur': "میں جَوی ہوں، ایک وائس اسسٹنٹ پروگرام جو آپ کو مختلف کاموں میں مدد کرنے کے لئے ڈیزائن کیا گیا ہے۔",
                     'bn': "আমি জয়ে، একটি ভয়েস অ্যাসისტ্যান্ট প্রোগ্রাম যা আপনাকে বিভিন্ন কাজে সাহায্য করার জন্য ডিজাইন করা হয়েছে।" # Added Bengali
                 }
                  speak(responses_general_about.get(response_lang, responses_general_about['en']), response_lang)


        elif intent == "ask_location":
            location = get_location() # Placeholder
            responses = {
                'en': f"Based on available information, you appear to be in {location}.",
                'hi': f"دستیاب جانکاری کے مطابق، آپ {location} میں प्रतीत होते ہیں۔", # Corrected Hindi/Urdu mix
                'es': f"Según la información disponible, pareces estar en {location}.",
                'ur': f"دستیاب معلومات کے مطابق، آپ {location} میں نظر آتے ہیں۔",
                'bn': f"উপलब्ধ তথ্য অনুযায়ী، আপনি {location} এ আছেন বলে মনে হচ্ছে।" # Added Bengali
            }
            speak(responses.get(response_lang, responses['en']), response_lang)

        elif intent == "tell_time":
            now = datetime.now()
            current_time = now.strftime("%I:%M %p") # e.g., 03:30 PM
            responses = {
                'en': f"The current time is {current_time}.",
                'hi': f"ابھی {current_time} بجے ہیں۔",
                'es': f"La hora actual es {current_time}.",
                'ur': f"موجودہ وقت {current_time} ہے۔",
                'bn': f"এখন সময় {current_time}।" # Added Bengali
            }
            speak(responses.get(response_lang, responses['en']), response_lang)

        elif intent == "ask_weather":
            weather_data = get_weather() # Placeholder
            if weather_data:
                 # Provide temperature in both Celsius and Fahrenheit
                 weather_text_en = f"The weather in {weather_data['location']} is {weather_data['condition']} with a temperature of {weather_data['temp_c']} degrees Celsius or {weather_data['temp_f']} degrees Fahrenheit."
                 weather_text_hi = f"{weather_data['location']} में मौसम {weather_data['condition']} है और temperature {weather_data['temp_c']} degrees Celsius or {weather_data['temp_f']} degrees Fahrenheit है।" # Corrected Hindi/Urdu mix
                 weather_text_es = f"El clima en {weather_data['location']} está {weather_data['condition']} con una temperatura de {weather_data['temp_c']} grados Celsius o {weather_data['temp_f']} grados Fahrenheit."
                 weather_text_ur = f"{weather_data['location']} میں موسم {weather_data['condition']} ہے اور درجہ حرارت {weather_data['temp_c']} ڈگری سیلسیس یا {weather_data['temp_f']} ڈگری فارن ہائیٹ ہے۔"
                 weather_text_bn = f"{weather_data['location']} এর আবহাওয়া {weather_data['condition']} এবং তাপমাত্রা {weather_data['temp_c']} ডিগ্রি সেলসিয়াস বা {weather_data['temp_f']} ডিগ্রি ফারেনহাইট।" # Added Bengali

                 weather_responses = {
                      'en': weather_text_en,
                      'hi': weather_text_hi, # Ensure the Hindi response is purely Hindi
                      'es': weather_text_es,
                      'ur': weather_text_ur,
                      'bn': weather_text_bn
                 }
                 # Use get_language_code for robust lookup
                 speak(weather_responses.get(get_language_code(response_lang), weather_responses['en']), response_lang) # Fallback handling

            else:
                 responses = {
                      'en': "Sorry, I couldn't get the weather information at the moment.",
                      'hi': "माफ़ करना, मुझे अभी मौसम کی جانکاری نہیں مل پائی۔", # Corrected Hindi/Urdu mix
                      'es': "Lo siento, no pude obtener la información del clima en este momento.",
                      'ur': "معاف کرنا، مجھے فی الحال मौसम کی معلومات نہیں مل سکی۔", # Corrected Hindi/Urdu mix
                      'bn': "দুঃখিত، আমি এই মুহূর্তে আবহাওয়ার তথ্য পেতে পারিনি।" # Added Bengali
                 }
                 speak(responses.get(response_lang, responses['en']), response_lang)


        elif intent == "translate":
             # This branch is for the *general* translate intent matched by TF-IDF ("translate this", "translate now")
             # The specific regex for "translate X to Y" is handled earlier.
             responses = {
                  'en': "What would you like me to translate and to which language?",
                  'hi': "आप क्या ترجمہ کرنا چاہیں گے اور کس زبان میں؟", # Corrected Hindi/Urdu mix
                  'es': "¿Qué te gustaría que tradujera y a qué idioma?",
                  'ur': "آپ کیا ترجمہ کرنا چاہیں گے اور کس زبان میں؟",
                  'bn': "আপনি কি অনুবাদ করতে চান এবং কোন ভাষায়?" # Added Bengali
             }
             speak(responses.get(response_lang, responses['en']), response_lang)


      


        elif intent == "unknown":
            # Handle unknown intent
            responses = {
                'en': "Sorry, I didn't understand that. Could you please rephrase?",
                'hi': "माफ़ करना, मुझे यह समझ نہیں آیا۔ کیا آپ کر پیا اسے دوبارہ کہہ سکتے ہیں؟", # Corrected Hindi/Urdu mix
                'es': "Lo siento, no entendí eso. ¿Podrías decirlo de otra manera?",
                'ur': "معاف کرنا، مجھے یہ سمجھ نہیں آیا۔ کیا آپ براہ کرم اسے دوبارہ کہہ سکتے ہیں؟",
                'bn': "দুঃখিত، আমি এটা বুঝতে পারিনি। আপনি কি দয়া করে অন্যভাবে বলতে পারেন?" # Added Bengali
            }
            speak(responses.get(response_lang, responses['en']), response_lang)

        # Optional: Add a small delay to prevent rapid loops if listen returns empty quickly
        time.sleep(0.1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting Joey.")
    finally:
        # Ensure mixer is fully quit on exit
        if pygame.mixer.get_init():
             pygame.mixer.quit()
             print("[INFO] Pygame mixer quit.")
        print("Joey has shut down.")