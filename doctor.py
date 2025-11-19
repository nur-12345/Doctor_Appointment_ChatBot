# doctor_fixed.py
import streamlit as st
import uuid
from dotenv import load_dotenv
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

# Database modules
from database import (
    register_user, login_user, insert_personal_information,
    fetch_user_chat_sessions, fetch_chat_history, save_chat_history,
    fetch_available_slots, book_appointment, collect_feedback
)

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# ---------- Custom CSS for Professional UI ----------
def inject_custom_css():
    st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    
    .main-header {
        color: #2c3e50;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid #3498db;
        margin-bottom: 2rem;
    }
    
    .user-message {
        background-color: #e3f2fd;
        padding: 12px 16px;
        border-radius: 18px 18px 0 18px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #bbdefb;
    }
    
    .assistant-message {
        background-color: black;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 0;
        margin: 8px 0;
        max-width: 80%;
        margin-right: auto;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    
    .stButton button {
        background-color: #3498db;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    .stButton button:hover {
        background-color: #2980b9;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 16px;
        color: #155724;
        margin: 16px 0;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 16px;
        color: #856404;
        margin: 16px 0;
    }
    
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 8px;
        padding: 16px;
        color: #0c5460;
        margin: 16px 0;
    }
    
    .session-button {
        width: 100%;
        margin: 5px 0;
        text-align: left;
        padding: 10px;
        border-radius: 5px;
        background-color: #34495e;
        color: white;
        border: none;
        transition: all 0.3s;
    }
    
    .session-button:hover {
        background-color: #3498db;
    }
    
    /* Fix for form submission */
    .stForm {
        margin-bottom: 20px;
    }
    
    /* Chat input styling */
    .chat-input {
        position: fixed;
        bottom: 20px;
        left: 20px;
        right: 20px;
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
        z-index: 1000;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------- Fixed Utility: safe rerun ----------
def safe_rerun():
    """
    Fixed rerun function that works with current Streamlit versions
    """
    try:
        # For newer Streamlit versions
        st.rerun()
    except Exception:
        try:
            # For older versions
            st.experimental_rerun()
        except Exception:
            # Final fallback - just set a flag and let Streamlit handle it
            st.session_state['_need_rerun'] = True
            st.stop()

# ---------- Chroma client & FAQ loading ----------
def read_faq_file(file_path):
    faq_data = {}
    current_question = None
    current_answer = []
    try:
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
    except FileNotFoundError:
        st.warning("FAQ file not found. FAQ-based answers will be disabled.")
    return faq_data

file_path = './faq.txt'
faq_data = read_faq_file(file_path)

# ---------- Toxicity model guard ----------
toxicity_model = None  # Simplified for now to avoid import issues

def detect_toxicity(text):
    if toxicity_model is None or not text:
        return False
    try:
        results = toxicity_model(text)
        return any(r.get('label','').lower().startswith('toxic') and r.get('score',0)>0.5 for r in results)
    except Exception:
        return False

# ---------- FAQ search ----------
def get_most_relevant_faq(user_input):
    if not faq_data:
        return None
    # Simple keyword matching as fallback
    user_input_lower = user_input.lower()
    for question, answer in faq_data.items():
        question_lower = question.lower()
        # Check if any significant words match
        question_words = set(question_lower.split())
        input_words = set(user_input_lower.split())
        common_words = question_words.intersection(input_words)
        if len(common_words) >= 2:  # At least 2 common words
            return question, answer, 0.8  # Fixed similarity score
    return None

# ---------- Build message list for AI ----------
def build_message_list_for_groq():
    messages = [{"role": "system", "content": """You are a helpful medical assistant. 
    Provide clear, concise, and helpful responses about healthcare and appointments. 
    Do not provide medical diagnoses. 
    If someone asks about booking an appointment, guide them to use the booking feature.
    Only respond to medical questions with appropriate information."""}]
    
    # Add conversation history (last 6 exchanges to avoid context overflow)
    chat_history = st.session_state.get('messages', [])
    messages.extend(chat_history[-12:])  # Keep last 6 exchanges (12 messages)
    
    return messages

# ---------- Generate AI response ----------
def generate_response(messages):
    if ChatGroq is None or not groq_api_key:
        return "AI service is currently unavailable. Please try again later."
    try:
        chat = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key, temperature=0.5)
        out = chat.invoke(messages)
        if hasattr(out, "content"):
            return out.content
        if isinstance(out, dict) and 'content' in out:
            return out['content']
        return str(out)
    except Exception as e:
        return f"I apologize, but I'm having trouble responding right now. Please try again."

# ---------- Process User Input ----------
def process_user_input(user_input):
    """Centralized function to process all user inputs"""
    if not user_input or not user_input.strip():
        return None
    
    user_input = user_input.strip()
    
    # Check for toxicity
    if detect_toxicity(user_input):
        return "I'm sorry, but I cannot respond to inappropriate language. Please rephrase your question."
    
    # Check for appointment booking intent
    booking_keywords = ['book appointment', 'schedule appointment', 'make appointment', 'want to book']
    if any(keyword in user_input.lower() for keyword in booking_keywords):
        st.session_state.page = 'booking'
        return "I can help you book an appointment. Please use the booking section to select your preferred date and time."
    
    # Try FAQ first
    faq_result = get_most_relevant_faq(user_input)
    if faq_result and len(faq_result) == 3:
        faq_question, faq_answer, similarity = faq_result
        if similarity > 0.7:
            return faq_answer
    
    # Use AI for general responses
    st.session_state.messages.append({"role": "user", "content": user_input})
    messages_for_ai = build_message_list_for_groq()
    
    with st.spinner("Thinking..."):
        ai_response = generate_response(messages_for_ai)
    
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
    return ai_response

# ---------- Initialize Session State ----------
def initialize_session_state():
    """Initialize all required session state variables"""
    if 'initialized' not in st.session_state:
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
            "page": 'chat',
            "initialized": True
        }
        
        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v

# ---------- Main Application ----------
def main():
    # Initialize session state
    initialize_session_state()
    
    # Inject custom CSS
    inject_custom_css()
    
    # Header
    st.markdown("""
   
    <style>
    .main-header h1 span {
    display: inline-block;
    white-space: nowrap;
    color: #ffffff; /* अपनी पसंद का color */
    }
    .main-header h1 {
    display:flex;
    align-items:center;
    justify-content:center;
    gap:10px;
    margin:0;
   }
   </style>

    <div class="main-header">
     <h1>
        <span style="font-size:1.5em;">🏥</span>
        <span>Doctor Appointment Chatbot</span>
     </h1>
     <p style="margin:0; font-size:1.1em; color:#7f8c8d;">
        Your virtual healthcare assistant
     </p>
    </div>
    """, unsafe_allow_html=True)


    # Login/Register Section
    if not st.session_state.get('username'):
        show_auth_section()
        return

    # Main Application (After Login)
    show_main_application()

def show_auth_section():
    """Show authentication section (login/register)"""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.write("### Welcome to Our Medical Service")
        st.write("Please login or register to continue")
        st.markdown('</div>', unsafe_allow_html=True)
        
        mode = st.radio("Select Mode", ("Login", "Register"), horizontal=True, key="auth_mode")
        
        if mode == "Register":
            show_register_form()
        else:
            show_login_form()

def show_register_form():
    """Show registration form"""
    st.write("## Create New Account")
    with st.form("register_form", clear_on_submit=True):
        username = st.text_input("Username", key="register_username")
        password = st.text_input("Password", type='password', key="register_password")
        confirm_password = st.text_input("Confirm Password", type='password', key="register_confirm_password")
        submitted = st.form_submit_button("Register")
        
        if submitted:
            handle_registration(username, password, confirm_password)

def handle_registration(username, password, confirm_password):
    """Handle user registration"""
    if not username or not password:
        st.markdown('<div class="warning-box">Please fill in all fields</div>', unsafe_allow_html=True)
    elif password != confirm_password:
        st.markdown('<div class="warning-box">Passwords do not match</div>', unsafe_allow_html=True)
    else:
        try:
            register_user(username, password)
            st.markdown('<div class="success-box">Registration successful! Please login.</div>', unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f'<div class="warning-box">Registration failed: {str(e)}</div>', unsafe_allow_html=True)

def show_login_form():
    """Show login form"""
    st.write("## Login to Your Account")
    with st.form("login_form", clear_on_submit=True):
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type='password', key="login_password")
        submitted = st.form_submit_button("Login")
        
        if submitted:
            handle_login(username, password)

def handle_login(username, password):
    """Handle user login"""
    if not username or not password:
        st.markdown('<div class="warning-box">Please enter both username and password</div>', unsafe_allow_html=True)
    else:
        if login_user(username, password):
            # Initialize user session
            st.session_state.username = username
            st.session_state.current_session_id = str(uuid.uuid4())
            st.session_state.past = []
            st.session_state.generated = []
            st.session_state.messages = [{"role": "assistant", "content": f"Hello {username}, welcome! How can I assist you today?"}]
            st.session_state.personal_info_collected = False
            safe_rerun()
        else:
            st.markdown('<div class="warning-box">Invalid username or password</div>', unsafe_allow_html=True)

def show_main_application():
    """Show main application after login"""
    # Sidebar
    show_sidebar()
    
    # Personal Information Collection
    if not st.session_state.personal_info_collected:
        show_personal_info_form()
        return
    
    # Main Chat Interface
    show_chat_interface()

def show_sidebar():
    """Show application sidebar"""
    with st.sidebar:
        st.markdown(f"""
        <div style="background:#3498db; color:white; padding:15px; border-radius:10px; margin-bottom:20px;">
            <h3 style="margin:0;">👤 {st.session_state.username}</h3>
            <p style="margin:0; font-size:0.9em;">Active Session</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Your Chat Sessions")
        try:
            user_sessions = fetch_user_chat_sessions(st.session_state.username)
        except Exception:
            user_sessions = []
            
        if user_sessions:
            for session_id in user_sessions:
                display_id = session_id[:8] + "..." if len(session_id) > 8 else session_id
                if st.button(f"📝 {display_id}", key=f"session_{session_id}"):
                    load_chat_session(session_id)
        else:
            st.info("No previous sessions")
            
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True, key="logout_btn"):
            handle_logout()

def load_chat_session(session_id):
    """Load a specific chat session"""
    st.session_state.current_session_id = session_id
    st.session_state.past = []
    st.session_state.generated = []
    try:
        history = fetch_chat_history(st.session_state.username, session_id)
        for user_msg, ai_msg in history:
            st.session_state.past.append(user_msg)
            st.session_state.generated.append(ai_msg)
    except Exception as e:
        st.error(f"Error loading chat history: {e}")
    safe_rerun()

def handle_logout():
    """Handle user logout"""
    try:
        collect_feedback(st.session_state.username)
    except Exception:
        pass
    
    # Clear session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    safe_rerun()

def show_personal_info_form():
    """Show personal information collection form"""
    st.markdown("""
    <div style="background:#e8f4f8; padding:20px; border-radius:10px; margin-bottom:20px;">
        <h2 style="color:#2c3e50; margin-top:0;">Complete Your Profile</h2>
        <p style="color:#7f8c8d;">We need some basic information to provide you with the best service.</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("personal_info_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full Name", key="personal_name")
        with col2:
            birth_date = st.date_input("Date of Birth", key="personal_dob")
            
        reason_for_appointment = st.text_area("Reason for Appointment", 
                                             placeholder="Please describe your symptoms or reason for seeking medical attention...",
                                             key="personal_reason")
        
        submitted = st.form_submit_button("Save Information", use_container_width=True)
        
        if submitted:
            if name and reason_for_appointment:
                save_personal_info(name, birth_date, reason_for_appointment)
            else:
                st.markdown('<div class="warning-box">Please fill in all required fields</div>', unsafe_allow_html=True)

def save_personal_info(name, birth_date, reason_for_appointment):
    """Save personal information"""
    try:
        insert_personal_information(st.session_state.username, name, birth_date, reason_for_appointment)
        st.session_state.personal_info_collected = True
        st.session_state.messages = [{"role": "assistant", "content": "Thank you! Your information has been saved. How can I assist you today?"}]
        safe_rerun()
    except Exception as e:
        st.error(f"Error saving personal information: {e}")

def show_chat_interface():
    """Show main chat interface"""
    # Display chat history
    show_chat_history()
    
    # Handle current page state
    if st.session_state.page == 'chat':
        show_chat_input()
    elif st.session_state.page == 'booking':
        show_booking_interface()

def show_chat_history():
    """Display chat history"""
    chat_container = st.container()
    with chat_container:
        for i, (user_msg, ai_msg) in enumerate(zip(st.session_state.past, st.session_state.generated)):
            st.markdown(f'<div class="user-message"><strong>You:</strong> {user_msg}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="assistant-message"><strong>Doctor:</strong> {ai_msg}</div>', unsafe_allow_html=True)

def show_chat_input():
    """Show chat input form"""
    st.markdown("---")
    
    # Quick action buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📅 Book Appointment", use_container_width=True, key="book_btn"):
            st.session_state.page = 'booking'
            safe_rerun()
    with col2:
        if st.button("🔄 New Session", use_container_width=True, key="new_session_btn"):
            st.session_state.current_session_id = str(uuid.uuid4())
            st.session_state.past = []
            st.session_state.generated = []
            st.session_state.messages = [{"role": "assistant", "content": "Starting new conversation. How can I help you?"}]
            safe_rerun()
    with col3:
        if st.button("ℹ️ Get Help", use_container_width=True, key="help_btn"):
            help_response = "I can help you with medical information, appointment booking, and general health questions. What would you like to know?"
            st.session_state.past.append("Get Help")
            st.session_state.generated.append(help_response)
            safe_rerun()
    
    # Chat input form
    with st.form(key='chat_form', clear_on_submit=True):
        user_input = st.text_input("Type your message here...", key="chat_input", 
                                 placeholder="Ask about symptoms, book appointment, or general health questions...")
        
        col1, col2 = st.columns([3, 1])
        with col2:
            submitted = st.form_submit_button("Send", use_container_width=True)
        
        if submitted and user_input:
            process_chat_input(user_input)

def process_chat_input(user_input):
    """Process user input in chat"""
    ai_response = process_user_input(user_input)
    
    if ai_response:
        # Update chat history
        st.session_state.past.append(user_input)
        st.session_state.generated.append(ai_response)
        
        # Save to database
        try:
            save_chat_history(
                st.session_state.username, 
                st.session_state.current_session_id, 
                user_input, 
                ai_response
            )
        except Exception as e:
            st.error(f"Error saving chat: {e}")
        
        safe_rerun()

def show_booking_interface():
    """Show appointment booking interface"""
    st.markdown("""
    <div style="background:#e8f4f8; padding:20px; border-radius:10px; margin-bottom:20px;">
        <h2 style="color:#2c3e50; margin-top:0;">📅 Book an Appointment</h2>
        <p style="color:#7f8c8d;">Select your preferred date and time for the appointment.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        appointment_date = st.date_input("Appointment Date", key='booking_date')
    with col2:
        if appointment_date:
            try:
                available_slots = fetch_available_slots(appointment_date)
                if available_slots:
                    slot_options = [slot.strftime("%H:%M") for slot in available_slots]
                    selected_slot = st.selectbox("Available Time Slots", slot_options, key="time_slot")
                else:
                    st.info("No available slots for the selected date.")
                    selected_slot = None
            except Exception as e:
                st.error(f"Error fetching available slots: {e}")
                selected_slot = None
    
    # Action buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("Confirm Booking", use_container_width=True, key="confirm_booking"):
            if appointment_date and selected_slot:
                handle_booking(appointment_date, selected_slot)
            else:
                st.warning("Please select both date and time slot.")
    
    with col2:
        if st.button("Back to Chat", use_container_width=True, key="back_to_chat"):
            st.session_state.page = 'chat'
            safe_rerun()
    
    with col3:
        if st.button("Check Availability", use_container_width=True, key="check_avail"):
            safe_rerun()

def handle_booking(appointment_date, slot_time_str):
    """Handle appointment booking"""
    try:
        result = book_appointment(st.session_state.username, appointment_date, slot_time_str)
        cleaned_result = str(result).strip().lower()
        
        if "already booked" in cleaned_result:
            st.markdown('<div class="warning-box">This slot is already booked. Please choose another time.</div>', unsafe_allow_html=True)
        elif "success" in cleaned_result:
            st.markdown(f'<div class="success-box">Appointment booked successfully for {appointment_date} at {slot_time_str}!</div>', unsafe_allow_html=True)
            st.session_state.appointment_booked = True
            st.session_state.page = 'chat'
            # Add booking confirmation to chat history
            confirmation_msg = f"Appointment booked for {appointment_date} at {slot_time_str}"
            st.session_state.past.append("Book appointment")
            st.session_state.generated.append(confirmation_msg)
            safe_rerun()
        else:
            st.error(f"Booking failed: {result}")
    except Exception as e:
        st.error(f"An error occurred during booking: {e}")

# Run the application
if __name__ == "__main__":
    main()