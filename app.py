import streamlit as st
import os
import tempfile
from backend import speech_to_text, prompt_to_image, classifier_reve
import matplotlib.pyplot as plt
import mimetypes


# ----------------------
# Config Streamlit
# ----------------------
st.set_page_config(page_title="Synthétiseur de Rêves", layout="wide")

if "historiques" not in st.session_state:
    st.session_state.historiques = []


# ----------------------
# Fonctions d'interface
# ----------------------

def create_headers_app():
    st.title(" Synthétiseur de Rêves")
    st.markdown("Partage ton rêve à l'oral, et laisse l'IA l’interpréter en image et émotions.")


def upload_audio():
    st.markdown("## 1. Enregistre ou uploade ton rêve")
    uploaded = st.file_uploader("Télécharger un fichier audio, types acceptés (.mp3, .wav, .m4a)", type=["mp3", "wav", "m4a"])
    mimes_accepts = {
        ".mp3": "audio/mp3"

    }
    if uploaded:
        filename = uploaded.name
        ext = os.path.splitext(filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(uploaded.read())
            return tmp.name
    return None


def print_transcription(audio_path):
    with st.spinner(" Transcription du rêve en cours..."):
        texte = speech_to_text(audio_path)
    st.markdown("## 2. Transcription du rêve")
    st.text_area("Texte du rêve", texte, height=200)
    return texte


def display_image(texte):
    st.markdown("## 3. Génération d'une image du rêve")
    with st.spinner(" Création de l'image à partir du texte..."):
        _ = prompt_to_image(prompt=texte)
    
    img_name = sorted([f for f in os.listdir("images") if f.endswith(".png")])[-1]
    st.image(f"images/{img_name}", caption="Image générée", use_container_width=True)
    return f"images/{img_name}"


def plot_sentiments(texte):
    st.markdown("## 4. Analyse émotionnelle du rêve")
    with st.spinner(" Analyse des émotions..."):
        sentiments = classifier_reve(texte)
    fig, ax = plt.subplots()
    ax.pie(sentiments.values(), labels=sentiments.keys(), autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)
    return sentiments


def save_consult_historics():
    st.markdown("## 5. Historique des rêves (session en cours)")
    if not st.session_state.historiques:
        st.info("Aucun rêve traité pour le moment.")
        return
    
    for idx, item in enumerate(reversed(st.session_state.historiques)):
        with st.expander(f"Rêve #{len(st.session_state.historiques) - idx}"):
            st.text_area("Texte du rêve", item["texte"], height=100)
            st.image(item["image"], use_container_width=True)
            fig, ax = plt.subplots()
            ax.pie(item["sentiments"].values(), labels=item["sentiments"].keys(), autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)


# Main Application 
def main():
    create_headers_app()

    audio_path = upload_audio()

    if audio_path:

        texte = print_transcription(audio_path)
        image_path = display_image(texte)
        sentiments = plot_sentiments(texte)

        st.session_state.historiques.append({
            "texte": texte,
            "sentiments": sentiments,
            "image": image_path
        })

        st.success(" Rêve traité avec succès !")

    save_consult_historics()


if __name__ == "__main__":
    main()
