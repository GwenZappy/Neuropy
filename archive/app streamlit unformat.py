import pyttsx3
import threading
import openai
from vosk import Model, KaldiRecognizer
import sounddevice as sd
import queue
import json
import streamlit as st

# API Key
from config import OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY

# Initialize Vosk Model
model_path = "/Users/Administrator/T540_HHO/vosk-model-small-en-us-0.15"
model = Model(model_path)
recognizer = KaldiRecognizer(model, 16000)

# Queue for streaming audio
q = queue.Queue()

# Initialize pyttsx3 for TTS
engine = pyttsx3.init(driverName='nsss')  # Use 'nsss' for macOS
speak_lock = threading.Lock()

def speak(text):
    """Thread-safe TTS with logging."""
    def tts():
        with speak_lock:
            print(f"Speaking: {text}")
            engine.say(text)
            engine.runAndWait()
            print("Finished Speaking.")
    
    tts_thread = threading.Thread(target=tts)
    tts_thread.start()

def audio_callback(indata, frames, time, status):
    """Callback to receive audio data."""
    if status:
        st.error(f"Stream Error: {status}")
    q.put(bytes(indata))

def analyze_sentiment_with_chatgpt(text):
    """Analyze sentiment and primary emotion."""
    prompt = f"""
    Analyze the sentiment of the following text based on the circumplex model of emotions. Identify the primary emotion (Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger, Anticipation) and its intensity (Low, Medium, High):\n\n{text}
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0
    )
    return response["choices"][0]["message"]["content"]

def extract_entities_with_emotions(text):
    """Extract entities and emotions with associations."""
    prompt = f"""
    Extract the following categories of entities from the text below. For each category, list the relevant details:

    1. **People**: Names of people mentioned in the text.
    2. **Locations**: Specific locations mentioned (e.g., park, apartment, city, bookstore).
    3. **Events**: Key actions or activities described in the text (e.g., walking, kissing, watching a movie).
    4. **Environment Conditions**: Any details about the surroundings or environment (e.g., weather, physical settings).
    5. **Emotions**: Identify emotions based on the circumplex model of emotions (Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger, Anticipation) and their intensity (Low, Medium, High).
    6. **Associations**: For each emotion, provide:
       - People
       - Locations
       - Events
       - Environment Conditions

    Text:
    {text}

    Please provide the output in a structured format.
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0
    )
    return response["choices"][0]["message"]["content"]

def process_transcription_with_chatgpt(transcription):
    """Process transcription text with sentiment analysis and entity extraction."""
    st.write("### Transcription")
    st.write(transcription)

    # Sentiment Analysis
    try:
        sentiment = analyze_sentiment_with_chatgpt(transcription)
        st.write("### Sentiment Analysis")
        st.write(sentiment)
    except Exception as e:
        st.error(f"Sentiment Analysis Error: {e}")

    # Entity Extraction with Emotions
    try:
        entities = extract_entities_with_emotions(transcription)
        st.write("### Extracted Entities and Emotions")
        st.write(entities)
    except Exception as e:
        st.error(f"Entity Extraction Error: {e}")

def continuous_transcription():
    """Continuously transcribe audio and process with ChatGPT."""
    speak("Tell me about your day.")
    st.info("Listening... (say 'that's it' to end)")
    full_transcription = ""
    with sd.InputStream(samplerate=16000, channels=1, dtype="int16", callback=audio_callback):
        try:
            while True:
                data = q.get()
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "")
                    st.write(f"Partial Transcript: {text}")
                    full_transcription += " " + text

                    if "that's it" in text.lower():
                        st.success("Stopping transcription...")
                        speak("Thanks for sharing!")
                        break
        except Exception as e:
            st.error(f"Error during transcription: {e}")
        finally:
            process_transcription_with_chatgpt(full_transcription)

# Streamlit App
def main():
    st.title("Home Hub - Conversational Assistant")
    st.write("This Home Hub listens to you, analyzes your emotions, and responds empathetically.")
    
    if st.button("Start Conversation"):
        try:
            continuous_transcription()
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()