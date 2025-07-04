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
    if not os.path.exists(file):
        raise FileNotFoundError(f"Le fichier introuuvable {file_path}")

    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    # Open the audio file
    with open(file_path, "rb") as file:
        # Create a transcription of the audio file
        transcription = client.audio.transcriptions.create(
        file=file, # Required audio file
        model="whisper-large-v3-turbo", # Required model to use for transcription
        prompt="Extrait de l'audio le texte, de la façon la plus consise et factuelle que tu peux",  # Optional
        response_format="verbose_json",  # Optional
        timestamp_granularities = ["word", "segment"], # Optional (must set response_format to "json" to use and can specify "word", "segment" (default), or both)
        language=language,  # Optional
        temperature=0.5 # Optional # 50% factuelle et 50% créative
    )

    return transcription.text

# Normalisation des resultats des proba à ce que la somme des proba des caractéristiques de l'utilisateur soit 1 (softmax)
def normalize_sentiment(sentiments):
    exp_values = {k: math.exp(v) for k, v in sentiments.items() if isinstance(v, (int, float))}
    
    total = sum(exp_values.values()) or 1e-10 # La somme des expo

    return {key: (value/total) for key, value in exp_values.items()}

# Classification du rêve
def classifier_reve(text):
    api_key = os.environ["MISTRAL_API_KEY"]
    mistral_model = "mistral-small-latest"
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
    try:
        chat_response = client.chat.complete(
            model = mistral_model,
            messages = messages,
            response_format = {
                "type": "json_object",
            }
        )
        result = json.loads(chat_response.choices[0].message.content)
        
        return normalize_sentiment(result)
    except Exception as e:
        print(f"Erreur API Mistral : {e}")
        return {
            "triste": 0.0,
            "heureux": 0.0,
            "en_colere": 0.0,
            "fatigue": 0.0,
            "anxieux": 0.0,
            "neutre": 0.0
        }


# Prompt to image -> display(image, video)
def prompt_to_image(prompt):
    url = 'https://clipdrop-api.co/text-to-image/v1'
    headers = { 'x-api-key': os.environ["CLIPDROP_API_KEY"]}
    files = {
        'prompt': (None, f"Génère moi ce rêve text ci-dessous en image sympa, immersive et factuelle: {prompt}", 'text/plain')
    }
    try:
        response = requests.post(url=url, headers=headers, files=files)
        if (response.ok):
            # Optionally save the image to a file
            time_temp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{time_temp}_reve_generated.png"
            
            os.makedirs("images", exist_ok=True)
            with open(f"images/{filename}", 'wb') as img_file:
                img_file.write(response.content)
            return response.headers   # heasers content the consumed & renaiming credits API Key

        else:
            response.raise_for_status()
    except Exception as e:
        print(f"Erreur API Clipdrop: {e}")
        return None




