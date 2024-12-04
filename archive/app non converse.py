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
    prompt = f"Analyze the sentiment of the following text based on the circumplex model of emotions. Identify the primary emotion (Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger, Anticipation) and its intensity (Low, Medium, High):\n\n{text}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Or "gpt-4" for higher accuracy
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0
    )
    return response["choices"][0]["message"]["content"]

def extract_entities_with_emotions(text):
    prompt = f"""
    Extract the following categories of entities from the text below. For each category, list the relevant details:

    1. **People**: Names of people mentioned in the text.
    2. **Locations**: Specific locations mentioned (e.g., park, apartment, city, bookstore).
    3. **Events**: Key actions or activities described in the text (e.g., walking, kissing, watching a movie).
    4. **Environment Conditions**: Any details about the surroundings or environment (e.g., weather, physical settings).
    5. **Emotions**: Identify emotions based on the circumplex model of emotions (Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger, Anticipation) and their intensity (Low, Medium, High). Specify the intensity using related words from the model (e.g., Serenity, Joy, Ecstasy).
    6. **Associations**: For each emotion, provide:
       - People: List all people associated with the emotion.
       - Locations: List all locations associated with the emotion.
       - Events: List all events associated with the emotion.
       - Environment Conditions: List all environmental conditions associated with the emotion.

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
    print("\nTranscription:")
    print(transcription)

    # Sentiment Analysis
    try:
        sentiment = analyze_sentiment_with_chatgpt(transcription)
        print("\nSentiment Analysis:")
        print(sentiment)
    except OpenAIError as e:
        print("Sentiment Analysis Error:", e)

    # Entity Extraction with Emotions
    try:
        entities = extract_entities_with_emotions(transcription)
        print("\nExtracted Entities and Emotions:")
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