# audio_processing.py
import speech_recognition as sr
from gtts import gTTS
import streamlit as st
import base64

def record_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening...")
        audio_data = recognizer.listen(source)
        st.success("Audio recorded!")
    return audio_data

# Function to recognize speech from audio
def recognize_speech(audio_data):
    recognizer = sr.Recognizer()
    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return "Sorry, I could not understand the audio."
    except sr.RequestError:
        return "Sorry, the speech recognition service is unavailable."

# Function to convert text to speech
def text_to_speech(text):
    tts = gTTS(text)
    return tts

# Function to save audio and generate download link
def generate_audio_download_link(audio, filename="output.mp3"):
    audio.save(filename)
    with open(filename, "rb") as f:
        audio_bytes = f.read()
    b64 = base64.b64encode(audio_bytes).decode()
    href = f'<a href="data:audio/mp3;base64,{b64}" download="{filename}">Download Audio</a>'
    return href
