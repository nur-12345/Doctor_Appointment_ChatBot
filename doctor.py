# doctor_fixed.py
import streamlit as st
import uuid
from dotenv import load_dotenv
from gtts import gTTS
import speech_recognition as sr
from io import BytesIO
import os
import re

# Optional / heavy imports guarded so app can still start if they fail
try:
    import chromadb
except Exception:
    chromadb = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# Groq / LangChain related imports - keep guarded
try:
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage, AIMessage
except Exception:
    ChatGroq = None
    HumanMessage = None
    AIMessage = None

# Database/audio modules (assumed present)
from database import (
    register_user, login_user, insert_personal_information,
    fetch_user_chat_sessions, fetch_chat_history, save_chat_history,
    fetch_available_slots, book_appointment, collect_feedback
)
from audio_processing import record_audio, recognize_speech, text_to_speech, generate_audio_download_link

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# ---------- Utility: safe rerun ----------
def safe_rerun():
    """
    Robust rerun function: tries st.experimental_rerun(), else raises Streamlit's
    internal rerun exception if available. If neither is available, set a
    session flag to request front-end refresh.
    """
    try:
        # Preferred / common API
        st.experimental_rerun()
    except AttributeError:
        # Try internal exception (works for many streamlit versions)
        try:
            from streamlit.runtime.scriptrunner.script_runner import RerunException
            raise RerunException()
        except Exception:
            # Final fallback: set a flag; the app cannot force a rerun on some builds
            st.session_state['_need_rerun'] = True

# ---------- Audio helper ----------
def play_text_audio(text):
    """
    Convert text to speech and play in Streamlit. Accepts text string.
    """
    if not text:
        return
    try:
        tts = gTTS(text=text, lang='en')
        audio_file = BytesIO()
        tts.write_to_fp(audio_file)
        st.audio(audio_file.getvalue(), format="audio/mp3")
    except Exception as e:
        # Don't crash the app if audio fails
        st.warning(f"Audio playback failed: {e}")

# ---------- Chroma client & FAQ loading ----------
def read_faq_file(file_path):
    faq_data = {}
    current_question = None
    current_answer = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            if line.startswith("Q:"):
                if current_question is not None:
                    faq_data[current_question] = " ".join(current_answer).strip()
                current_question = line[2:].strip()
                current_answer = []
            elif line.startswith("A:"):
                current_answer.append(line[2:].strip())
            else:
                current_answer.append(line)
        if current_question is not None:
            faq_data[current_question] = " ".join(current_answer).strip()
    return faq_data

file_path = './faq.txt'
faq_data = {}
try:
    faq_data = read_faq_file(file_path)
except FileNotFoundError:
    st.warning("FAQ file not found. FAQ-based answers will be disabled.")

def create_chromadb_client():
    if chromadb is None:
        return None
    try:
        client = chromadb.Client()
        return client
    except Exception:
        return None

client = create_chromadb_client()
collection_name = 'faq_collection'

def initialize_collection(client, collection_name, faq_data):
    if client is None:
        return None
    try:
        # API may vary between Chromadb versions
        try:
            collection = client.get_collection(name=collection_name)
        except Exception:
            # some clients use different param names
            collection = client.create_collection(name=collection_name)
        # Add documents (embedding placeholder if no real embedding)
        for q, a in faq_data.items():
            embedding = [0.0] * 128
            try:
                collection.add(ids=[q], metadatas=[{"question": q}], documents=[a], embeddings=[embedding])
            except Exception:
                # fallback to older signature
                collection.add({'id': q, 'embedding': embedding, 'text': a})
        return collection
    except Exception:
        return None

collection = initialize_collection(client, collection_name, faq_data)

# ---------- Toxicity model guard ----------
try:
    from transformers import pipeline
    toxicity_model = pipeline("text-classification", model="unitary/toxic-bert")
except Exception:
    toxicity_model = None

def detect_toxicity(text):
    if toxicity_model is None:
        return False
    try:
        results = toxicity_model(text)
        return any(r.get('label','').lower().startswith('toxic') and r.get('score',0)>0.5 for r in results)
    except Exception:
        return False

# ---------- FAQ search ----------
def get_most_relevant_faq(user_input, collection_obj):
    if collection_obj is None or not faq_data:
        return None
    if SentenceTransformer is None:
        return None
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode(user_input).tolist()
        try:
            results = collection_obj.query(embeddings=[query_embedding], n_results=5)
        except TypeError:
            # alternative signature
            results = collection_obj.query(query_embeddings=[query_embedding], n_results=5)
        # results structure may vary; try common access patterns
        documents = results.get('documents') if isinstance(results, dict) else None
        distances = results.get('distances') if isinstance(results, dict) else None
        if documents and len(documents) > 0:
            faq_question = documents[0][0] if isinstance(documents[0], list) else documents[0]
            faq_answer = results['documents'][0][0] if isinstance(results['documents'][0], list) else results['documents'][0]
            similarity = distances[0][0] if distances else 0.0
            return faq_question, faq_answer, similarity
        return None
    except Exception:
        return None

# ---------- Build message list for AI ----------
def build_message_list_for_groq():
    # return a list of dicts for ChatGroq
    base = [{"role": "system", "content": "You are a helpful assistant."}]
    base.extend(st.session_state.get('messages', []))
    return base

# ---------- Generate AI response ----------
def generate_response(messages):
    if ChatGroq is None:
        return "AI backend not configured."
    try:
        chat = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key, temperature=0.5)
        out = chat.invoke(messages)
        # Accept different return shapes
        if hasattr(out, "content"):
            return out.content
        if isinstance(out, dict) and 'content' in out:
            return out['content']
        return str(out)
    except Exception as e:
        return f"AI generation failed: {e}"

# ---------- Streamlit session state defaults ----------
defaults = {
    "username": None,
    "current_session_id": None,
    "past": [],
    "generated": [],
    "input_text": "",
    "messages": [{"role": "assistant", "content": "Welcome to the doctor appointment service! How can I assist you today?"}],
    "personal_info_collected": False,
    "feedback_collected": False,
    "logout": False,
    "appointment_booked": False,
    "page": 'chat'
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------- UI / Logic ----------
# Navbar simplified
st.markdown("<h1>Doctor Appointment Chatbot</h1>", unsafe_allow_html=True)

if not st.session_state.get('username'):
    mode = st.radio("Select Mode", ("Login", "Register"))
    if mode == "Register":
        st.write("## Register")
        username = st.text_input("Enter Username", key="register_username")
        password = st.text_input("Enter Password", type='password', key="register_password")
        confirm_password = st.text_input("Confirm Password", type='password', key="register_confirm_password")
        if st.button("Register"):
            if password == confirm_password:
                register_user(username, password)
                st.success("Registered. Please login.")
            else:
                st.warning("Passwords do not match")
    else:
        st.write("## Login")
        username = st.text_input("Enter Username", key="login_username")
        password = st.text_input("Enter Password", type='password', key="login_password")
        if st.button("Login"):
            if login_user(username, password):
                st.success(f"Logged in as {username}")
                st.session_state['username'] = username
                st.session_state['current_session_id'] = str(uuid.uuid4())
                st.session_state['past'] = []
                st.session_state['generated'] = []
                st.session_state['messages'] = []
                st.session_state['greeting'] = f"Hello {username}, welcome! How can I assist you?"
                safe_rerun()
            else:
                st.warning("Invalid Username/Password")

if st.session_state.get('username'):
    st.sidebar.write("## Your Chat Sessions:")
    try:
        user_sessions = fetch_user_chat_sessions(st.session_state['username'])
    except Exception:
        user_sessions = []

    if not st.session_state["personal_info_collected"]:
        st.write("## Please Provide Your Personal Information")
        name = st.text_input("Name", key="personal_name")
        birth_date = st.date_input("Birth Date", key="personal_dob")
        reason_for_appointment = st.text_area("Reason for Appointment", key="personal_reason")
        if st.button("Submit Personal Information"):
            insert_personal_information(st.session_state["username"], name, birth_date, reason_for_appointment)
            st.session_state["personal_info_collected"] = True
            st.success("Personal information saved successfully.")
            st.session_state['greeting'] = "You can now start chatting with the doctor."
            safe_rerun()

    for session_id in user_sessions:
        if st.sidebar.button(session_id):
            st.session_state['current_session_id'] = session_id
            st.session_state['past'] = []
            st.session_state['generated'] = []
            history = fetch_chat_history(st.session_state['username'], session_id)
            for user_msg, ai_msg in history:
                st.session_state['past'].append(user_msg)
                st.session_state['generated'].append(ai_msg)

    # Chat UI
    chat_container = st.container()
    with chat_container:
        # greeting
        if 'greeting' in st.session_state:
            st.write(st.session_state['greeting'])
            del st.session_state['greeting']

        # show history
        for user_msg, ai_msg in zip(st.session_state['past'], st.session_state['generated']):
            st.write(f"**You:** {user_msg}")
            st.write(f"**Doctor:** {ai_msg}")

    if st.session_state['page'] == 'chat':
        with st.form(key='chat_form', clear_on_submit=True):
            col1, col2 = st.columns([9, 1])
            with col1:
                user_input = st.text_input("You:", key="input_text_form")
            with col2:
                submit_button = st.form_submit_button(label="↑")

            if submit_button and user_input:
                if st.session_state['current_session_id'] is None:
                    st.error("No active session selected.")
                elif detect_toxicity(user_input):
                    st.warning("Your input contains inappropriate language. Please modify your message.")
                    ai_response = "Sorry, I cannot respond to this request."
                    st.session_state['generated'].append(ai_response)
                    play_text_audio(ai_response)
                elif "book appointment" in user_input.lower():
                    st.session_state['page'] = 'booking'
                    safe_rerun()
                else:
                    # Try FAQ first
                    result = get_most_relevant_faq(user_input, collection)
                    if result and len(result) == 3:
                        faq_question, faq_answer, similarity = result
                        similarity_threshold = 0.7
                        if similarity > similarity_threshold:
                            ai_response = faq_answer
                            play_text_audio(ai_response)
                        else:
                            # Call LLM as fallback
                            st.session_state['messages'].append({"role": "user", "content": user_input})
                            with st.spinner("Generating response..."):
                                ai_response = generate_response(build_message_list_for_groq())
                            play_text_audio(ai_response)
                    else:
                        st.session_state['messages'].append({"role": "user", "content": user_input})
                        with st.spinner("Generating response..."):
                            ai_response = generate_response(build_message_list_for_groq())
                        play_text_audio(ai_response)

                    st.session_state['past'].append(user_input)
                    st.session_state['generated'].append(ai_response)
                    # attempt to save; ignore DB errors
                    try:
                        save_chat_history(st.session_state['username'], st.session_state['current_session_id'], user_input, ai_response)
                    except Exception:
                        pass

                    with chat_container:
                        st.write(f"**You:** {user_input}")
                        st.write(f"**Doctor:** {ai_response}")

    # Booking page
    if st.session_state['page'] == 'booking':
        st.write("Please select a date for the appointment:")
        appointment_date = st.date_input("Appointment Date", key='booking_date')
        if appointment_date:
            available_slots = fetch_available_slots(appointment_date)
            if available_slots:
                slot_time_str = st.selectbox("Select a time slot", [slot.strftime("%H:%M") for slot in available_slots])
                if st.button("Confirm Booking"):
                    try:
                        result = book_appointment(st.session_state['username'], appointment_date, slot_time_str)
                        cleaned_result = str(result).strip().lower().rstrip('.')
                        if cleaned_result == "slot already booked":
                            st.warning("This slot is already booked. Please choose another available slot.")
                        elif cleaned_result == "appointment booked successfully":
                            st.session_state['generated'].append(result)
                            st.session_state['appointment_booked'] = True
                            st.success(result)
                            st.session_state['page'] = 'chat'
                            safe_rerun()
                        else:
                            st.error(f"Unexpected result: {result}. Please try again.")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")
            else:
                st.write("No available slots for the selected date.")

    if st.button("Logout"):
        st.session_state["logout"] = True

    if st.session_state["logout"]:
        try:
            collect_feedback(st.session_state["username"])
        except Exception:
            pass
        st.session_state["feedback_collected"] = True

    if st.session_state["feedback_collected"]:
        # Clear sensitive session state and show message
        st.session_state.update({k: defaults[k] for k in defaults})
        st.success("You have been logged out.")
        safe_rerun()





# import streamlit as st
# from langchain_openai import ChatGroq
# from langchain.schema import HumanMessage, AIMessage
# import uuid
# from dotenv import load_dotenv
# from gtts import gTTS
# import speech_recognition as sr
# from io import BytesIO
# import os
# import chromadb
# from chromadb.config import Settings
# from sentence_transformers import SentenceTransformer
# import re
# from transformers import pipeline

# from database import (
#     register_user, login_user, insert_personal_information,
#     fetch_user_chat_sessions, fetch_chat_history, save_chat_history,
#     fetch_available_slots, book_appointment, collect_feedback
# )
# from audio_processing import record_audio, recognize_speech, text_to_speech, generate_audio_download_link

# load_dotenv()

# # Initialize Groq API
# groq_api_key = os.getenv("GROQ_API_KEY")

# def audio():
#     tts = gTTS(text=ai_response, lang='en')
#     audio_file = BytesIO()
#     tts.write_to_fp(audio_file)
#     st.audio(audio_file.getvalue(), format="audio/mp3")

# # Create ChromaDB client
# def create_chromadb_client():
#     try:
#         client = chromadb.Client()
#         print("ChromaDB client created successfully")
#         return client
#     except Exception as e:
#         print(f"Error creating ChromaDB client: {e}")
#         return None

# client = create_chromadb_client()
# collection_name = 'faq_collection'

# # Read FAQ data
# def read_faq_file(file_path):
#     faq_data = {}
#     current_question = None
#     current_answer = []

#     with open(file_path, 'r') as file:
#         for line in file:
#             line = line.strip()
#             if line.startswith("Q:"):
#                 if current_question is not None:
#                     faq_data[current_question] = " ".join(current_answer)
#                 current_question = line
#                 current_answer = []
#             elif line.startswith("A:"):
#                 current_answer.append(line[2:].strip())
#             else:
#                 current_answer.append(line)
#         if current_question is not None:
#             faq_data[current_question] = " ".join(current_answer)
#     return faq_data

# file_path = './faq.txt'
# faq_data = read_faq_file(file_path)

# # Initialize ChromaDB collection
# def initialize_collection(client, collection_name, faq_data):
#     try:
#         try:
#             collection = client.get_collection(collection_name)
#             print(f"Collection '{collection_name}' already exists.")
#         except ValueError:
#             print(f"Collection '{collection_name}' not found. Creating a new one.")
#             collection = client.create_collection(collection_name)

#         documents = []
#         for question, answer in faq_data.items():
#             embedding = [0.0] * 128
#             document = {
#                 'id': question,
#                 'text': answer,
#                 'embedding': embedding
#             }
#             documents.append(document)

#         for document in documents:
#             collection.add({
#                 'id': document['id'],
#                 'embedding': document['embedding'],
#                 'text': document['text']
#             })

#         return collection

#     except Exception as e:
#         return None

# collection = initialize_collection(client, collection_name, faq_data)

# # Initialize toxicity detection model
# toxicity_model = pipeline("text-classification", model="unitary/toxic-bert")

# def detect_toxicity(text):
#     results = toxicity_model(text)
#     return any(result['label'] == 'toxic' and result['score'] > 0.5 for result in results)

# def recognize_speech():
#     recognizer = sr.Recognizer()
#     with sr.Microphone() as source:
#         recognizer.adjust_for_ambient_noise(source)
#         st.info("Listening... Speak now.")
#         audio = recognizer.listen(source)
#     try:
#         return recognizer.recognize_google(audio)
#     except sr.UnknownValueError:
#         return "Sorry, I could not understand the audio."
#     except sr.RequestError:
#         return "Sorry, the speech recognition service is unavailable."

# def text_to_speech(text):
#     tts = gTTS(text)
#     return tts

# # Streamlit session state initialization
# if "username" not in st.session_state:
#     st.session_state["username"] = None
# if "current_session_id" not in st.session_state:
#     st.session_state["current_session_id"] = None
# if "past" not in st.session_state:
#     st.session_state["past"] = []
# if "generated" not in st.session_state:
#     st.session_state["generated"] = []
# if "messages" not in st.session_state:
#     st.session_state["messages"] = []
# if "personal_info_collected" not in st.session_state:
#     st.session_state["personal_info_collected"] = False
# if "feedback_collected" not in st.session_state:
#     st.session_state["feedback_collected"] = False
# if "logout" not in st.session_state:
#     st.session_state["logout"] = False
# if 'appointment_booked' not in st.session_state:
#     st.session_state['appointment_booked'] = False
# if 'page' not in st.session_state:
#     st.session_state['page'] = 'chat'

# def extract_date_time(user_input):
#     date_time_pattern = r"(\d{4}-\d{2}-\d{2}) at (\d{2}:\d{2})"
#     match = re.search(date_time_pattern, user_input)
#     if match:
#         date_str, time_str = match.groups()
#         return date_str, time_str
#     return None, None

# def generate_response(messages):
#     chat = ChatGroq(
#         model="Gemma2-9b-It",
#         groq_api_key=groq_api_key,
#         temperature=0.5
#     )

#     ai_response = chat.invoke(messages)
#     return ai_response.content

# def get_most_relevant_faq(user_input, collection):
#     if collection is None:
#         return None

#     model = SentenceTransformer('all-MiniLM-L6-v2')
#     query_embedding = model.encode(user_input).tolist()

#     try:
#         results = collection.query(
#             embeddings=[query_embedding],
#             k=5
#         )

#         if results and len(results['documents']) > 0:
#             most_relevant = results['documents'][0]
#             similarity = results['distances'][0]
#             faq_question = most_relevant['id']
#             faq_answer = most_relevant['text']

#             return faq_question, faq_answer, similarity
#         return None

#     except Exception as e:
#         return None

# def build_message_list():
#     messages = []
#     for user_msg in st.session_state['past']:
#         messages.append(HumanMessage(content=user_msg))
#     for ai_msg in st.session_state['generated']:
#         messages.append(AIMessage(content=ai_msg))
#     return messages

# def logout_user():
#     st.session_state.clear()
#     st.success("You have successfully logged out.")

# # Navbar HTML and CSS
# navbar_html = """
# <nav class="navbar">
#   <div class="navbar-container">
#     <h1 class="navbar-title">Doctor Appointment Chatbot</h1>
#   </div>
# </nav>
# """

# navbar_css = """
# <style>
# .navbar { background-color: black; padding: 5px; }
# .navbar-title { color: white; text-align: center; font-size: 30px; margin: 0; }
# </style>
# """

# st.markdown(navbar_html, unsafe_allow_html=True)
# st.markdown(navbar_css, unsafe_allow_html=True)

# if not st.session_state.get('username'):
#     mode = st.radio("Select Mode", ("Login", "Register"))

#     if mode == "Register":
#         st.write("## Register")
#         username = st.text_input("Enter Username", key="register_username")
#         password = st.text_input("Enter Password", type='password', key="register_password")
#         confirm_password = st.text_input("Confirm Password", type='password", key="register_confirm_password")

#         if st.button("Register"):
#             if password == confirm_password:
#                 register_user(username, password)
#             else:
#                 st.warning("Passwords do not match")

#     elif mode == "Login":
#         st.write("## Login")
#         username = st.text_input("Enter Username", key="login_username")
#         password = st.text_input("Enter Password", type='password', key="login_password")

#         if st.button("Login"):
#             if login_user(username, password):
#                 st.success(f"Logged in as {username}")
