import openai
from openai import OpenAIError
from vosk import Model, KaldiRecognizer
import sounddevice as sd
import queue
import json


# API Key
from config import OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY

# Initialize Vosk Model
model_path = "/Users/Administrator/T540_HHO/vosk-model-small-en-us-0.15"  
model = Model(model_path)
recognizer = KaldiRecognizer(model, 16000)

# Queue for streaming audio
q = queue.Queue()

def audio_callback(indata, frames, time, status):
    """Callback to receive audio data."""
    if status:
        print(f"Stream Error: {status}")
    q.put(bytes(indata))

def analyze_sentiment_with_chatgpt(text):
    prompt = f"Analyze the sentiment of the following text and explain why. Provide the sentiment as Positive, Negative, or Neutral:\n\n{text}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Or "gpt-3.5-turbo" for lower cost
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        temperature=0
    )
    return response["choices"][0]["message"]["content"]

def extract_entities_with_chatgpt(text):
    prompt = f"Extract entities from the following text and categorize them as People, Dates, Emotions and things associated with the emotions:\n\n{text}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0
    )
    return response["choices"][0]["message"]["content"]

# Process Transcription with ChatGPT
def process_transcription_with_chatgpt(transcription):
    """Process transcription text with sentiment analysis and entity extraction."""
    print("\nTranscription:")
    print(transcription)

    # Sentiment Analysis
    try:
        sentiment = analyze_sentiment_with_chatgpt(transcription)
        print("\nSentiment Analysis:")
        print(sentiment)
    except OpenAIError as e:
        print("Sentiment Analysis Error:", e)

    # Entity Extraction
    try:
        entities = extract_entities_with_chatgpt(transcription)
        print("\nExtracted Entities:")
        print(entities)
    except OpenAIError as e:
        print("Entity Extraction Error:", e)

# Continuous Transcription
def continuous_transcription():
    """Continuously transcribe audio and process with ChatGPT."""
    print("Listening... (say 'stop' to end)")
    full_transcription = ""
    with sd.InputStream(samplerate=16000, channels=1, dtype="int16", callback=audio_callback):
        while True:
            data = q.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "")
                print(f"Partial Transcript: {text}")
                full_transcription += " " + text

                if "stop" in text.lower():
                    print("\nStopping transcription...")
                    break

    process_transcription_with_chatgpt(full_transcription)

# Main Function
if __name__ == "__main__":
    try:
        continuous_transcription()
    except Exception as e:
        print("Error:", e)