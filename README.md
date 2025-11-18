# Doctor_Appointment_ChatBot
AI-powered Doctor Appointment Chatbot built with Streamlit, Groq LLM, ChromaDB, PostgreSQL, and Docker. Includes voice input, toxicity detection, FAQ retrieval, and smart appointment booking.


An AI-powered medical assistant built using Streamlit, Groq LLM, Speech Recognition, ChromaDB, PostgreSQL, and Docker, enabling users to register, chat with an AI doctor, and book appointments through text or voice.


This project integrates NLP, vector search, toxicity detection, appointment management, and DevOps deployment.

        
    📌 Features Overview
     1. User System
       Register & Login
       Secure password hashing (bcrypt)
       Chat sessions stored in DB
       Personal information storage (name, age, reason)
       
    2. Smart Chatbot
     Chat with Groq Llama-3.1 (fallback when no FAQ match)
     FAQ retrieval using ChromaDB + SentenceTransformer embeddings
     Toxicity detection using Unitary Toxic-BERT
     Text + Voice input supported
     gTTS voice responses + audio download option
     
    3. Appointment Booking
     Real-time available slot generation
     9 AM – 5 PM slots (30-min interval)
     Lunch break auto-blocked (1 PM – 2 PM)
     Prevents double-booking
     Saves booking info to PostgreSQL
     
    4. Feedback System
     Users can submit feedback at logout
     Stored in database
     
    5. DevOps Integration
    Dockerized application
    Dockerfile included
    docker-compose.yml included
    Supports seamless deployment on cloud servers
    
    📁 Project Structure
     .
     ├── doctor.py                  # Main Streamlit app (chatbot + UI)
     ├── database.py                # All PostgreSQL database operations
     ├── audio_processing.py        # Voice recording, STT, TTS
     ├── Dockerfile                 # Container for full project
     ├── docker-compose.yml         # Multi-container orchestration (if added)
     ├── faq.txt                    # FAQ dataset for ChromaDB
     ├── requirements.txt           # Project dependencies
     └── README.md                  # Documentation
     
  ⚙️ Installation & Setup
    1. Clone the repository
    
    git clone https://github.com/yourusername/doctor-appointment-chatbot.git
    cd doctor-appointment-chatbot
  2. Create and activate virtual environment
     
    python -m venv venv
    source venv/bin/activate      # macOS/Linux
    venv\Scripts\activate         # Windows
   3. Install dependencies
      
    pip install -r requirements.txt
    
  4. Setup Environment Variables

    Create a .env file:
    GROQ_API_KEY=your_groq_key_here
    DB_HOST=your_host
    DB_USER=your_user
    DB_PASSWORD=your_password
    DB_NAME=your_dbname
    DB_PORT=5432
    
  ▶️ Running the Application
  Without Docker
  
    streamlit run doctor.py
    
  With Docker
    Build image:
    
    docker build -t doctor-chatbot .
  Run container:
  
    docker run -p 8501:8501 doctor-chatbot
  Using docker-compose:
  
    docker-compose up --build
     
  🧠 How the Chatbot Works
  
   1. Input Handling
      
     ✔ Text input
    ✔ Audio input (SpeechRecognition)
    ✔ Transcription → Google STT
    
   2. Safety Layer
           ✔ Toxicity detection with unitary/toxic-bert
           ✔ Blocks abusive queries
   3. FAQ Retrieval
          User message → Embedding
          ChromaDB similarity search
    If similarity ≥ threshold → return FAQ answer

  5. Groq LLM Response

    If no similar FAQ found
    Fallback to llama-3.1-8b-instant via Groq API
 6. Voice Output
      gTTS generates MP3
      Downloadable audio response available
    
💾 Database Operations (PostgreSQL)

    Tables Created
    Users
    Chat history
    Appointment slots
    User feedback
    Personal information
    Core Functions
    Register/Login
    Insert personal details
    Save chat messages
    Get chat history
    Generate available appointment slots
    Book appointment
    Save feedback
    
🔊 Audio Processing Details
audio_processing.py handles:

Recording voice using Microphone
Converting speech to text
Generating voice response (gTTS)
Returning audio download link

📦 Tech Stack

    Component	Technology
    UI	Streamlit
    Speech	SpeechRecognition, gTTS
    Database	PostgreSQL (psycopg2)
    NLP	Groq Llama 3.1
    Vector DB	ChromaDB
    Toxicity	Unitary Toxic-BERT
    ML	Sentence Transformers
    Deployment	Docker, docker-compose

📜 License
This project is open for learning and development use.
