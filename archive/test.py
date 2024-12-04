import openai

from config import OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY

def extract_entities_with_emotion_model(text):
    """Extract entities and emotions using the circumplex emotion model."""
    prompt = f"""
    Extract the following categories of entities from the text below. For each category, list the relevant details:

    1. **People**: Names of people mentioned in the text.
    2. **Locations**: Specific locations mentioned (e.g., park, apartment, city, bookstore).
    3. **Events**: Key actions or activities described in the text (e.g., walking, kissing, watching a movie).
    4. **Environment Conditions**: Any details about the surroundings or environment (e.g., weather, physical settings).
    5. **Emotions**: Identify emotions based on the circumplex model of emotions (Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger, Anticipation) and their intensity (low, medium, high). Specify the intensity using related words from the model (e.g., Serenity, Joy, Ecstasy).
    6. **Associations**: For each emotion, provide the following:
       - People: List all people associated with the emotion.
       - Locations: List all locations associated with the emotion.
       - Events: List all events associated with the emotion.
       - Environment Conditions: List all environmental conditions associated with the emotion.

    Format the output like this:

    1. **People**:
       - [Name1]
       - [Name2]

    2. **Locations**:
       - [Location1]
       - [Location2]

    3. **Events**:
       - [Event1]
       - [Event2]

    4. **Environment Conditions**:
       - [Condition1]

    5. **Emotions**:
       - [Emotion1]: [Intensity]
       - [Emotion2]: [Intensity]

    6. **Associations**:
       - **[Emotion1]**:
         - Intensity: [Low/Medium/High]
         - People: [Person1], [Person2]
         - Locations: [Location1], [Location2]
         - Events: [Event1], [Event2]
         - Environment Conditions: [Condition1]
       - **[Emotion2]**:
         - Intensity: [Low/Medium/High]
         - People: [Person3], [Person4]
         - Locations: [Location3], [Location4]
         - Events: [Event3], [Event4]
         - Environment Conditions: [Condition2]

    Text:
    {text}

    Please ensure each category is clearly labeled, and provide N/A if there is no data for a subcategory.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Replace with gpt-4 if necessary
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0
    )
    return response["choices"][0]["message"]["content"]

# Test the function
text = """this weekend i hang out with logan and then it was pretty good because we walked around the park and then he just look handsome ah that day and he hugged me from behind in the bookstore afterwards we went back to his apartment and that we watch a movie and he starts kissing me and then we start making out after that i come home and i actually feel a bit regret because bike i shouldn't have done that a shouldn't have just given yeah so i'm bit worried now"""
entities = extract_entities_with_emotion_model(text)
print(entities)