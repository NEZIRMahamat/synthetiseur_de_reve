# Import Librairies
from datetime import date, datetime
import math
import os
import re
import json
import time
import requests
from groq import Groq
from dotenv import load_dotenv
from mistralai import Mistral


load_dotenv(".env")

# Lecteur de fichier
def read_file(file_path):
    with open(file_path, "r") as file:
        return file.read()


# Trancription de l'audio en text : Audio to text -> print(result)
def speech_to_text(file_path, language="fr"):
    # Le cas où le fichier n'existe pas
    
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    # Open the audio file
    with open(file_path, "rb") as file:
        # Create a transcription of the audio file
        transcription = client.audio.transcriptions.create(
        file=file, # Required audio file
        model="whisper-large-v3-turbo", # Required model to use for transcription
        prompt="Extrait de l'audio le texte, de la façon la plus consise et la plus factuelle que tu peux",  # Optional
        response_format="verbose_json",  # Optional
        timestamp_granularities = ["word", "segment"], # Optional (must set response_format to "json" to use and can specify "word", "segment" (default), or both)
        language=language,  # Optional
        temperature=0.5 # Optional
    )

    return transcription.text

# Normalisation des resultats des proba à ce la somme des caractériistiqus de l'utilisateur vaut 1 (softmax)
def normalize_sentiment(sentiments):
    sentiments_normalized = {}
    exp_values = {k: math.exp(v) for k, v in sentiments.items() if isinstance(v, (int, float))}
    total = sum(exp_values.values())
    for key, value in exp_values.items():
        sentiments_normalized[key] = value / total if total != 0 else 0
    
    return sentiments_normalized

# Classification du rêve
def classifier_reve(text):
    api_key = os.environ.get("MISTRAL_API_KEY")
    model_0 = "mistral-small-latest"
    model = "mistral-large-latest"
    client = Mistral(api_key=api_key)
    messages = [
        {
            "role": "system",
            "content": read_file("./context.txt")
        },

        {
            "role": "user",
            "content": f"Analyse le texte suivant et envoie ta reponse uniquement au format JSON : {text}",
        }
    ]
    chat_response = client.chat.complete(
        model = model_0,
        messages = messages,
        response_format = {
            "type": "json_object",
        }
    )
    result = json.loads(chat_response.choices[0].message.content)
    
    return normalize_sentiment(result)


# Prompt to image -> display(image, video)
def prompt_to_image(prompt):
    url = 'https://clipdrop-api.co/text-to-image/v1'
    headers = { 'x-api-key': os.environ.get("CLIPDROP_API_KEY")}
    files = {
        'prompt': (None, f"Génère moi ce rêve text ci-dessous en image sympa, immersive et factuelle: {prompt}", 'text/plain')
    }
   
    response = requests.post(url=url, headers=headers, files=files)
    try:
        if (response.ok):
            # Optionally save the image to a file
            time_temp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{time_temp}_reve_generated.png"
            
            os.makedirs("images", exist_ok=True)
            with open(f"images/{filename}", 'wb') as img_file:
                 img_file.write(response.content)
            return response.headers   # contains the bytes of the returned image

        else:
            response.raise_for_status()
    except Exception as e:
        print(f"An error occurred: {e}")
        return None




