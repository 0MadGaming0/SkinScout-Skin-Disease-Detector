import streamlit as st
import torch
import pickle
from PIL import Image
from torchvision import transforms
from groq import Groq
from chatbot.chatbot import get_chatbot_response
import json
from datetime import datetime
import os
import streamlit.components.v1 as components

def auto_scroll():
    # Inject JavaScript to automatically scroll to the bottom of the main page and nested scrollable containers.
    components.html(
        """
        <script>
            setTimeout(function() {
                var parent_doc = window.parent.document;
                // Scroll the main page window to the bottom
                window.parent.window.scrollTo(0, parent_doc.body.scrollHeight);
                // Scroll all scrollable vertical containers to the bottom
                var divs = parent_doc.querySelectorAll('div[data-testid="stVerticalBlock"], div[style*="overflow"]');
                divs.forEach(function(div) {
                    if (div.scrollHeight > div.clientHeight) {
                        div.scrollTop = div.scrollHeight;
                    }
                });
            }, 50);
        </script>
        """,
        height=0,
    )

def markdown_to_html(text):
    # Convert **bold** to <strong>bold</strong>
    parts = text.split("**")
    html_parts = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            html_parts.append(f"<strong>{part}</strong>")
        else:
            html_parts.append(part)
    res = "".join(html_parts)
    # Convert newlines to <br>
    res = res.replace("\n", "<br>")
    return res

def render_chat_message(role, content):
    role_label = "user" if role == "user" else "chatbot"
    if role_label == "user":
        content_html = markdown_to_html(content)
        st.markdown(f'<div class="chat-message-row user-row"><span class="chat-label user-label">👤 user</span><span class="chat-text user-text">{content_html}</span></div>', unsafe_allow_html=True)
    else:
        disclaimer_marker = "⚠️ This is an AI-generated response"
        if disclaimer_marker in content:
            parts = content.split(disclaimer_marker, 1)
            main_text = parts[0].strip()
            disclaimer_text = disclaimer_marker + parts[1]
        else:
            main_text = content
            disclaimer_text = ""
            
        disclaimer_html = f'<div class="chat-disclaimer">{markdown_to_html(disclaimer_text)}</div>' if disclaimer_text else ""
        main_text_html = markdown_to_html(main_text)
        st.markdown(f'<div class="chat-message-row chatbot-row"><div><span class="chat-label chatbot-label">🤖 chatbot</span></div><div class="chatbot-content-wrapper"><span class="chat-text chatbot-text">{main_text_html}</span>{disclaimer_html}</div></div>', unsafe_allow_html=True)

# Load API Key from environment, .env file, or streamlit secrets
def load_groq_api_key():
    key = os.getenv("GROQ_API_KEY")
    if key:
        return key
    if os.path.exists(".env"):
        try:
            with open(".env", "r") as f:
                for line in f:
                    if line.startswith("GROQ_API_KEY="):
                        return line.strip().split("=", 1)[1].strip('"\'')
        except Exception:
            pass
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return ""

GROQ_API_KEY = load_groq_api_key()
client = Groq(api_key=GROQ_API_KEY)

st.set_page_config(page_title="SkinScout - Skin Disease Detection", layout="wide")

# Load custom styling design system
if os.path.exists("style.css"):
    with open("style.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load Model with CPU fallback
original_load = torch.load

def cpu_load(*args, **kwargs):
    kwargs["map_location"] = torch.device("cpu")
    return original_load(*args, **kwargs)

torch.load = cpu_load

with open("skin_disease_model.pkl", "rb") as f:
    model = pickle.load(f)

model.eval()

# Global Display Mapping
classes = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

display_names = {
    "akiec": "Actinic Keratoses",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevus",
    "vasc": "Vascular Lesion"
}

disease_info = {
    "akiec": "A rough, scaly patch on the skin caused by years of sun exposure (pre-cancerous).",
    "bcc": "A common type of skin cancer that usually develops on sun-exposed skin.",
    "bkl": "A harmless skin growth that often appears with aging (benign keratosis).",
    "df": "A benign skin nodule that usually appears on the legs or arms.",
    "mel": "A serious form of skin cancer that starts in pigment-producing cells (melanocytes).",
    "nv": "Common benign moles formed by pigment-producing cells (melanocytic nevus).",
    "vasc": "Skin abnormalities involving blood vessels, such as angiomas."
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Save chat history helper
def save_chat_history(disease, messages):
    history_file = "chat_history.json"
    
    # Track the active session timestamp to prevent duplicate logs during single conversation session
    if "session_timestamp" not in st.session_state:
        st.session_state.session_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    session_time = st.session_state.session_timestamp
    image_path = st.session_state.get("uploaded_image_path")
    
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                history = json.load(f)
        except Exception:
            history = []
    else:
        history = []
        
    # Find if an entry with this timestamp already exists
    found = False
    for entry in history:
        if entry.get("timestamp") == session_time:
            entry["messages"] = list(messages)
            entry["disease"] = disease
            if image_path:
                entry["image_path"] = image_path
            found = True
            break
            
    if not found:
        new_entry = {
            "timestamp": session_time,
            "disease": disease,
            "messages": list(messages)
        }
        if image_path:
            new_entry["image_path"] = image_path
        history.append(new_entry)
        
    with open(history_file, "w") as f:
        json.dump(history, f, indent=4)

def render_sidebar():
    st.sidebar.markdown("""
        <div style="text-align: center; padding-top: 10px; margin-bottom: 20px;">
            <span style="font-size: 40px; display: block; margin-bottom: 8px;">🏥</span>
            <h2 style="margin: 0; color: #0f172a; font-weight: 800; font-size: 20px; letter-spacing: -0.5px;">SkinScout AI</h2>
            <span style="color: #64748b; font-size: 12.5px; font-weight: 600;">Dermatology Assistant</span>
        </div>
        <hr style="margin: 0 0 20px 0; border: 0; border-top: 1px solid #e2e8f0;">
    """, unsafe_allow_html=True)



    st.sidebar.markdown("<br><b style='font-size: 13.5px; color: #475569; letter-spacing: -0.2px;'>Recent Consultations</b>", unsafe_allow_html=True)

    history_file = "chat_history.json"
    history = []
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                history = json.load(f)
        except Exception:
            history = []

    if history:
        # Reverse history to show latest first
        recent_items = list(reversed(history))
        for idx, item in enumerate(recent_items):
            h_idx = len(history) - 1 - idx
            disease_name = item.get("disease", "Unknown Condition")
            timestamp = item.get("timestamp", "N/A")
            
            # Format display name
            display_title = disease_name
            if len(display_title) > 20:
                display_title = display_title[:17] + "..."

            col_info, col_del = st.sidebar.columns([8.2, 1.8])
            
            with col_info:
                if st.button(f"💬 {display_title}", key=f"sidebar_resume_{h_idx}", use_container_width=True, help=f"{disease_name} ({timestamp})"):
                    st.session_state.session_timestamp = timestamp
                    messages = item.get("messages", [])
                    
                    if "General" in disease_name:
                        st.session_state.home_messages = list(messages)
                        st.session_state.chat_step = 6
                        if "uploaded_image" in st.session_state:
                            del st.session_state.uploaded_image
                        if "uploaded_image_path" in st.session_state:
                            del st.session_state.uploaded_image_path
                        st.session_state.page = "general_chatbot"
                    else:
                        is_scanner = False
                        if messages:
                            first_msg = messages[0].get("content", "")
                            if "analyzed your image" in first_msg:
                                is_scanner = True

                        inv_display_names = {v: k for k, v in display_names.items()}
                        disease_key = inv_display_names.get(disease_name, "akiec")

                        if is_scanner:
                            st.session_state.messages = list(messages)
                            st.session_state.disease = disease_key
                            st.session_state.confidence = 100.0
                            image_path = item.get("image_path")
                            if image_path and os.path.exists(image_path):
                                st.session_state.uploaded_image_path = image_path
                                if "uploaded_image" in st.session_state:
                                    del st.session_state.uploaded_image
                            else:
                                if "uploaded_image" in st.session_state:
                                    del st.session_state.uploaded_image
                                if "uploaded_image_path" in st.session_state:
                                    del st.session_state.uploaded_image_path
                            st.session_state.page = "result"
                        else:
                            st.session_state.symptoms_messages = list(messages)
                            st.session_state.symptoms_prediction = disease_name
                            st.session_state.symptoms_description = disease_info.get(disease_key, "No description available.")
                            st.session_state.symptoms_precautions = "• Keep the area clean\n• Avoid direct friction or itching\n• Consult a clinical dermatologist"
                            st.session_state.symptoms_analyzed = True
                            if "uploaded_image" in st.session_state:
                                del st.session_state.uploaded_image
                            if "uploaded_image_path" in st.session_state:
                                del st.session_state.uploaded_image_path
                            st.session_state.page = "symptoms"
                    st.rerun()
            
            with col_del:
                if st.button("🗑️", key=f"sidebar_del_{h_idx}", use_container_width=True, help=f"Delete thread: {disease_name}"):
                    # Remove image if it exists
                    img_path = item.get("image_path")
                    if img_path and os.path.exists(img_path):
                        try:
                            os.remove(img_path)
                        except Exception:
                            pass
                    history.pop(h_idx)
                    with open(history_file, "w") as f:
                        json.dump(history, f, indent=4)
                    
                    # Clear active session state if it matches the deleted thread
                    if st.session_state.get("session_timestamp") == timestamp:
                        for key in ["messages", "symptoms_messages", "home_messages", "chat_step", "chat_answers", "session_timestamp", "uploaded_image", "uploaded_image_path", "symptoms_analyzed", "disease", "confidence"]:
                            if key in st.session_state:
                                del st.session_state[key]
                        st.session_state.page = "home"
                    
                    st.success("Deleted!")
                    st.rerun()
    else:
        st.sidebar.info("No recent consultations.")

# Global Navbar Component
def render_header(page_title):
    if st.session_state.get("logged_in") and st.session_state.page != "login":
        col_title, col_user, col_hist, col_logout = st.columns([7.0, 1.0, 1.0, 1.0])
        with col_title:
            st.markdown(f"""
                <div class="nav-container-left">
                    <span style="font-size: 28px;">🏥</span>
                    <span style="font-weight: 800; font-size: 24px; color: #0f172a; letter-spacing: -0.5px;">SkinScout</span>
                    <span style="color: #cbd5e1; font-size: 18px; margin: 0 4px;">|</span>
                    <span style="font-weight: 600; font-size: 16px; color: #64748b; margin-top: 4px;">{page_title}</span>
                </div>
            """, unsafe_allow_html=True)
        with col_user:
            st.markdown(f"""
                <div style="text-align: right; display: flex; align-items: center; justify-content: flex-end; height: 100%;">
                    <span class="nav-user-badge">👤 {st.session_state.username}</span>
                </div>
            """, unsafe_allow_html=True)
        with col_hist:
            if st.button("📜 History", key="global_history_btn"):
                st.session_state.page = "history"
                st.rerun()
        with col_logout:
            if st.button("🚪 Logout", key="global_logout_btn"):
                st.session_state.logged_in = False
                st.session_state.page = "login"
                st.query_params.clear()
                if "session_timestamp" in st.session_state:
                    del st.session_state.session_timestamp
                st.rerun()
        st.markdown("<hr style='margin-top: 0; margin-bottom: 24px; border: 0; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)

# Session State Initialization
if "logged_in" not in st.session_state:
    if st.query_params.get("logged_in") == "true":
        st.session_state.logged_in = True
        st.session_state.username = st.query_params.get("username", "admin")
        st.session_state.page = "home"
    else:
        st.session_state.logged_in = False

if "page" not in st.session_state:
    st.session_state.page = "login"
    
if st.session_state.logged_in and st.session_state.page == "login":
    st.session_state.page = "home"

if st.session_state.get("logged_in") and st.session_state.page != "login":
    render_sidebar()


# ----------------------------------------------------
# 1. LOGIN PAGE
# ----------------------------------------------------
if st.session_state.page == "login":
    _, col, _ = st.columns([1, 1.8, 1])

    with col:
        # Marker element to target this column specifically in CSS
        st.markdown('<div class="login-card-marker"></div>', unsafe_allow_html=True)

        st.markdown("""
            <div class="login-header-container">
                <div class="login-logo">🏥</div>
                <h1 class="login-title">SkinScout</h1>
                <p class="login-subtitle">AI-Powered Dermatological Analysis</p>
            </div>
        """, unsafe_allow_html=True)

        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")

        if st.button("Sign In"):
            if username == "admin" and password == "1234":
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.page = "home"
                st.query_params["logged_in"] = "true"
                st.query_params["username"] = username
                st.rerun()
            else:
                st.error("Invalid username or password. Please try again.")


# ----------------------------------------------------
# 2. HOME PAGE (DASHBOARD)
# ----------------------------------------------------
elif st.session_state.page == "home":
    render_header("Home Dashboard")

    st.markdown(f"""
        <div style="margin-bottom: 40px; text-align: center;">
            <span style="font-size: 14px; font-weight: 700; color: #6366f1; text-transform: uppercase; letter-spacing: 1.5px; display: block; margin-bottom: 12px;">👋 Welcome back, {st.session_state.username}!</span>
            <h2 style="font-weight: 800; font-size: 30px; color: #0f172a; margin-bottom: 10px; letter-spacing: -0.5px;">Intelligent Skin Care Insights</h2>
            <p style="color: #475569; font-size: 16px; margin: 0; max-width: 620px; display: inline-block; line-height: 1.6;">
                Select a diagnostic module below to begin, or launch our interactive AI consultation assistant at the bottom.
            </p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card-marker-scanner"></div>', unsafe_allow_html=True)
        st.markdown("""
            <div class="card-icon-wrapper icon-scanner">📸</div>
            <h3 class="card-title">Image Diagnosis</h3>
            <p class="card-desc">Upload or capture a photo of a skin lesion for automated AI-based disease detection, confidence assessment, and follow-up consultation.</p>
        """, unsafe_allow_html=True)
        if st.button("Open Scanner", key="home_scanner_btn", use_container_width=True):
            st.session_state.page = "scanner"
            if "messages" in st.session_state:
                del st.session_state.messages
            if "uploaded_image" in st.session_state:
                del st.session_state.uploaded_image
            if "uploaded_image_path" in st.session_state:
                del st.session_state.uploaded_image_path
            if "session_timestamp" in st.session_state:
                del st.session_state.session_timestamp
            st.rerun()

    with col2:
        st.markdown('<div class="card-marker-symptoms"></div>', unsafe_allow_html=True)
        st.markdown("""
            <div class="card-icon-wrapper icon-symptoms">🩺</div>
            <h3 class="card-title">Symptoms Checker</h3>
            <p class="card-desc">Answer symptoms questionnaires without photos to analyze potential skin conditions, get home-care suggestions, and consult the AI.</p>
        """, unsafe_allow_html=True)
        if st.button("Check Symptoms", key="home_symptoms_btn", use_container_width=True):
            st.session_state.page = "symptoms"
            if "symptoms_analyzed" in st.session_state:
                del st.session_state.symptoms_analyzed
            if "symptoms_messages" in st.session_state:
                del st.session_state.symptoms_messages
            if "uploaded_image" in st.session_state:
                del st.session_state.uploaded_image
            if "uploaded_image_path" in st.session_state:
                del st.session_state.uploaded_image_path
            if "session_timestamp" in st.session_state:
                del st.session_state.session_timestamp
            st.rerun()

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Centered glowing neon chatbot button at the bottom center
    _, center_col, _ = st.columns([1.5, 1.4, 1.5])
    with center_col:
        st.markdown('<div class="chatbot-btn-marker"></div>', unsafe_allow_html=True)
        if st.button("💬 Any queries?", key="home_chatbot_btn", use_container_width=True):
            st.session_state.page = "general_chatbot"
            if "home_messages" in st.session_state:
                del st.session_state.home_messages
            if "chat_step" in st.session_state:
                del st.session_state.chat_step
            if "chat_answers" in st.session_state:
                del st.session_state.chat_answers
            if "session_timestamp" in st.session_state:
                del st.session_state.session_timestamp
            st.rerun()


# ----------------------------------------------------
# 3. AI GENERAL CHATBOT PAGE
# ----------------------------------------------------
elif st.session_state.page == "general_chatbot":
    render_header("AI Chat Assistant")

    col_back, _, col_clear = st.columns([1.6, 6.4, 2.0])
    with col_back:
        if st.button("⬅ Dashboard", key="chatbot_back_btn", use_container_width=True):
            st.session_state.page = "home"
            if "chat_step" in st.session_state:
                del st.session_state.chat_step
            if "chat_answers" in st.session_state:
                del st.session_state.chat_answers
            st.session_state.home_messages = []
            st.rerun()
    with col_clear:
        st.markdown('<div class="chat-clear-marker"></div>', unsafe_allow_html=True)
        if st.button("🧹 Clear Chat", key="chatbot_clear_btn", use_container_width=True):
            st.session_state.home_messages = []
            if "chat_step" in st.session_state:
                del st.session_state.chat_step
            if "chat_answers" in st.session_state:
                del st.session_state.chat_answers
            if "session_timestamp" in st.session_state:
                del st.session_state.session_timestamp
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Initialize guided interview states
    # Example:
    # user : hello
    # chatbot : hello what can i do for you
    if "home_messages" not in st.session_state or len(st.session_state.home_messages) == 0:
        st.session_state.home_messages = [
            {
                "role": "assistant",
                "content": "👋 Hello! I am your SkinScout Assistant. I am here to help answer any queries you have, or we can start a quick skin consultation."
            }
        ]
        st.session_state.chat_step = 0
        st.session_state.chat_answers = {}

    current_step = st.session_state.get("chat_step", 0)

    # Render steps dot container if still within assessment
    if current_step == 0:
        st.markdown("""
            <div class="chat-progress-container">
                <span class="chat-progress-label">🎯 SELECT CHAT MODE</span>
                <div style="font-size: 13.5px; font-weight: 700; color: #6366f1;">Choose an option or type below</div>
            </div>
        """, unsafe_allow_html=True)
    elif 1 <= current_step <= 5:
        dots_html = ""
        for i in range(1, 6):
            if i < current_step:
                dots_html += '<div class="chat-step-dot chat-step-dot-completed"></div>'
            elif i == current_step:
                dots_html += '<div class="chat-step-dot chat-step-dot-active"></div>'
            else:
                dots_html += '<div class="chat-step-dot"></div>'
        step_labels = {
            1: "Symptom Overview", 
            2: "Duration", 
            3: "Location", 
            4: "Sensations", 
            5: "Medical History"
        }
        st.markdown(f"""
            <div class="chat-progress-container">
                <span class="chat-progress-label">📋 GUIDED CONSULTATION: {step_labels[current_step]}</span>
                <div class="chat-progress-steps">{dots_html}</div>
            </div>
        """, unsafe_allow_html=True)
    elif current_step == 6:
        st.markdown("""
            <div class="chat-progress-container">
                <span class="chat-progress-label" style="color: #10b981; font-weight: 700;">✅ ASSESSMENT COMPLETED &mdash; Follow-up enabled</span>
                <div class="chat-progress-steps">
                    <div class="chat-step-dot chat-step-dot-completed"></div>
                    <div class="chat-step-dot chat-step-dot-completed"></div>
                    <div class="chat-step-dot chat-step-dot-completed"></div>
                    <div class="chat-step-dot chat-step-dot-completed"></div>
                    <div class="chat-step-dot chat-step-dot-completed"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Display messages
    for message in st.session_state.home_messages:
        render_chat_message(message["role"], message["content"])

    if current_step == 0:
        st.markdown("<div style='margin-top: 10px; margin-bottom: 10px; text-align: center; font-weight: 700; color: #475569;'>Choose how you would like to proceed:</div>", unsafe_allow_html=True)
        _, col_opt1, col_opt2, _ = st.columns([2, 1.5, 1.5, 2])
        with col_opt1:
            st.markdown('<div class="chat-choice-direct-marker"></div>', unsafe_allow_html=True)
            if st.button("💬 General Query", key="choose_direct_query", use_container_width=True):
                st.session_state.chat_step = 6
                st.session_state.home_messages.append({
                    "role": "assistant",
                    "content": "Sure! I am here to help. Feel free to ask me any queries or questions about skin health directly."
                })
                st.rerun()
        with col_opt2:
            st.markdown('<div class="chat-choice-guided-marker"></div>', unsafe_allow_html=True)
            if st.button("🩺 Start Consultation", key="choose_guided_consultation", use_container_width=True):
                st.session_state.chat_step = 1
                st.session_state.home_messages.append({
                    "role": "assistant",
                    "content": "**Question 1:** Are you having any type of skin symptoms (like a rash, mole, dry skin, or lesion)?"
                })
                st.rerun()

    # Chat input
    user_question = st.chat_input("Type your response or question here...", key="general_chat_input")

    if user_question:
        # Append user message
        st.session_state.home_messages.append({"role": "user", "content": user_question})

        if current_step == 0:
            # Check if user input is a query rather than starting the guided symptoms consultation
            user_question_lower = user_question.strip().lower()
            query_words = ["what", "how", "why", "explain", "tell me", "can you", "describe", "is it", "are you", "do you", "which", "where", "when", "please", "query", "question"]
            greetings = ["hello", "hi", "hey", "greetings", "yo", "morning", "afternoon", "evening"]
            
            is_query = (
                user_question_lower.endswith("?") or 
                any(w in user_question_lower for w in query_words) or
                user_question_lower in greetings or
                any(user_question_lower.startswith(g) for g in greetings)
            )
            
            if is_query:
                st.session_state.chat_step = 6
                with st.spinner("Thinking..."):
                    response = get_chatbot_response(
                        GROQ_API_KEY,
                        "General Skin Health (Direct Query)",
                        "Direct user query during assessment.",
                        user_question,
                        st.session_state.home_messages[:-1]
                    )
                st.session_state.home_messages.append({"role": "assistant", "content": response})
                save_chat_history("General Skin Assessment", st.session_state.home_messages)
                st.rerun()
            else:
                # Direct symptom input or start request
                start_keywords = ["start", "consultation", "questions", "begin", "guided", "yes", "ok", "okay", "sure", "yep", "yeah"]
                if user_question_lower in start_keywords or any(k in user_question_lower for k in ["start consultation", "guided check", "check skin"]):
                    st.session_state.chat_step = 1
                    st.session_state.home_messages.append({
                        "role": "assistant",
                        "content": "**Question 1:** Are you having any type of skin symptoms (like a rash, mole, dry skin, or lesion)?"
                    })
                    st.rerun()
                else:
                    # Treat their description as answer to Question 1!
                    st.session_state.chat_answers[1] = user_question
                    st.session_state.chat_step = 2
                    st.session_state.home_messages.append({
                        "role": "assistant",
                        "content": "**Question 2:** How long have you noticed these symptoms (e.g., days, weeks, months)?"
                    })
                    st.rerun()

        elif 1 <= current_step < 6:
            # Check if user input is a query rather than an answer to the current step
            user_question_lower = user_question.strip().lower()
            query_words = ["what", "how", "why", "explain", "tell me", "can you", "describe", "is it", "are you", "do you", "which", "where", "when", "please", "query", "question"]
            is_query = user_question_lower.endswith("?") or any(w in user_question_lower for w in query_words)
            
            if is_query:
                st.session_state.chat_step = 6
                with st.spinner("Thinking..."):
                    response = get_chatbot_response(
                        GROQ_API_KEY,
                        "General Skin Health (Direct Query)",
                        "Direct user query during assessment.",
                        user_question,
                        st.session_state.home_messages[:-1]
                    )
                st.session_state.home_messages.append({"role": "assistant", "content": response})
                save_chat_history("General Skin Assessment", st.session_state.home_messages)
                st.rerun()

            if current_step < 5:
                # Save response and proceed
                st.session_state.chat_answers[current_step] = user_question
                next_step = current_step + 1
                st.session_state.chat_step = next_step

                questions = {
                    2: "**Question 2:** How long have you noticed these symptoms (e.g., days, weeks, months)?",
                    3: "**Question 3:** Where on your body is the affected area located?",
                    4: "**Question 4:** Is the area painful, itchy, bleeding, or scaling?",
                    5: "**Question 5:** Have you had a history of skin allergies or similar conditions?"
                }

                st.session_state.home_messages.append({"role": "assistant", "content": questions[next_step]})
                st.rerun()

            elif current_step == 5:
                st.session_state.chat_answers[5] = user_question
                st.session_state.chat_step = 6

                with st.spinner("Analyzing your answers, please wait..."):
                    summary_prompt = f"""
                    The user has completed the skin symptoms questionnaire. Here are their answers:
                    - Question 1 (Symptoms): {st.session_state.chat_answers.get(1)}
                    - Question 2 (Duration): {st.session_state.chat_answers.get(2)}
                    - Question 3 (Location): {st.session_state.chat_answers.get(3)}
                    - Question 4 (Sensations): {st.session_state.chat_answers.get(4)}
                    - Question 5 (Medical/Allergy History): {st.session_state.chat_answers.get(5)}

                    Please analyze these symptoms educationally:
                    1. Provide a professional description of potential conditions that match these characteristics.
                    2. Give important precautions, daily skincare tips, or home-care adjustments.
                    3. Clearly specify when they should seek immediate professional medical evaluation.
                    """
                    response = get_chatbot_response(
                        GROQ_API_KEY,
                        "General Skin Symptoms Assessment",
                        "Skin symptom assessment questionnaire analysis.",
                        summary_prompt
                    )

                    st.session_state.home_messages.append({
                        "role": "assistant",
                        "content": f"📋 **Assessment Analysis Result**:\n\n{response}\n\n*Feel free to ask me any follow-up questions below.*"
                    })
                    save_chat_history("General Skin Assessment", st.session_state.home_messages)
                st.rerun()

        else:
            # Free-form follow-up conversation after assessment (current_step == 6)
            with st.spinner("Thinking..."):
                response = get_chatbot_response(
                    GROQ_API_KEY,
                    "General Skin Health (Follow-up)",
                    "Follow-up discussion after guided skin assessment.",
                    user_question,
                    st.session_state.home_messages[:-1]
                )
            st.session_state.home_messages.append({"role": "assistant", "content": response})
            save_chat_history("General Skin Assessment", st.session_state.home_messages)
            st.rerun()

    # Skip button trigger (shown only during guided consultation steps 1-5)
    if 1 <= current_step <= 5:
        st.markdown("<br>", unsafe_allow_html=True)
        col_skip_space, col_skip = st.columns([7.8, 2.2])
        with col_skip:
            if st.button("💬 Skip & Ask Direct Query", use_container_width=True):
                st.session_state.chat_step = 6
                st.session_state.home_messages = [
                    {
                        "role": "assistant",
                        "content": "Sure! I am here to help. Feel free to ask me any queries or questions about skin health directly."
                    }
                ]
                st.rerun()

    auto_scroll()


# ----------------------------------------------------
# 4. SCANNER PAGE (IMAGE SUBMISSION)
# ----------------------------------------------------
elif st.session_state.page == "scanner":
    render_header("Image Diagnosis Scanner")

    col_back, _ = st.columns([1.5, 8.5])
    with col_back:
        if st.button("⬅ Dashboard", key="scanner_back_btn", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # 1. Method Selector Card
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-card-title">📸 Choose Input Method</div>', unsafe_allow_html=True)
    input_method = st.radio(
        "Choose Input Method",
        ["Upload Image", "Capture Image"],
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # 2. File Input Panel
    st.markdown('<div class="panel-card" style="text-align: center;">', unsafe_allow_html=True)
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader(
            "Upload Skin Image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        camera_image = None
        if uploaded_file:
            st.image(uploaded_file, width=320)
    else:
        camera_image = st.camera_input("Capture Image")
        uploaded_file = None
        if camera_image:
            st.image(camera_image, width=320)
    st.markdown('</div>', unsafe_allow_html=True)

    # 3. Patient Details Card
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-card-title">📋 Patient & Lesion Details (Optional)</div>', unsafe_allow_html=True)
    col_age, col_dur, col_loc = st.columns(3)
    with col_age:
        age = st.number_input("Age", min_value=1, max_value=120, value=25, key="scanner_age")
    with col_dur:
        duration = st.selectbox(
            "How long have you had this lesion?",
            ["Less than 1 week", "1-4 weeks", "1-6 months", "More than 6 months"],
            key="scanner_duration"
        )
    with col_loc:
        location = st.selectbox(
            "Affected Body Part",
            ["Face", "Neck", "Arms", "Hands", "Chest", "Back", "Legs", "Feet", "Other"],
            key="scanner_location"
        )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Action Trigger
    st.markdown('<div class="main-action-btn">', unsafe_allow_html=True)
    if st.button("🔍 Run AI Detection", key="scanner_detect_btn"):
        if uploaded_file or camera_image:
            if "session_timestamp" in st.session_state:
                del st.session_state.session_timestamp
            if "messages" in st.session_state:
                del st.session_state.messages
            
            # Generate session timestamp immediately to serve as unique image filename
            st.session_state.session_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            image_source = uploaded_file if uploaded_file else camera_image
            image_obj = Image.open(image_source).convert("RGB")
            
            # Save upload or capture to history directory
            os.makedirs("history_images", exist_ok=True)
            image_path = f"history_images/{st.session_state.session_timestamp.replace(':', '-').replace(' ', '_')}.png"
            image_obj.save(image_path)
            st.session_state.uploaded_image_path = image_path
            
            image_tensor = transform(image_obj).unsqueeze(0)

            with st.spinner("Processing image and running predictions..."):
                with torch.no_grad():
                    outputs = model(image_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)

            st.session_state.disease = classes[predicted.item()]
            st.session_state.confidence = confidence.item() * 100
            st.session_state.uploaded_image = image_source
            
            # Initialize assistant's initial message and auto-save consultation to history
            display_name = display_names.get(st.session_state.disease, st.session_state.disease)
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": (
                        f"👋 I have analyzed your image and detected **{display_name}** (Confidence: {st.session_state.confidence:.2f}%).\n\n"
                        f"To help provide relevant context, could you share a few details about the lesion?\n"
                        f"1. **Is the spot painful, itchy, bleeding, or completely asymptomatic?**\n"
                        f"2. **Have you noticed any recent changes in its size, border shape, or color?**\n"
                        f"3. **Have you consulted a dermatologist about this spot before?**"
                    )
                }
            ]
            save_chat_history(display_name, st.session_state.messages)
            
            st.session_state.page = "result"
            st.rerun()
        else:
            st.warning("Please upload or capture an image first before classification.")
    st.markdown('</div>', unsafe_allow_html=True)


# ----------------------------------------------------
# 5. SYMPTOMS CHECKER PAGE
# ----------------------------------------------------
elif st.session_state.page == "symptoms":
    render_header("Symptoms Assessment Checker")

    col_back, _ = st.columns([1.5, 8.5])
    with col_back:
        if st.button("⬅ Dashboard", key="symptoms_back_btn", use_container_width=True):
            st.session_state.page = "home"
            if "symptoms_analyzed" in st.session_state:
                del st.session_state.symptoms_analyzed
            if "symptoms_messages" in st.session_state:
                del st.session_state.symptoms_messages
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Split workspace depending on active status
    is_analyzed = st.session_state.get("symptoms_analyzed")
    if is_analyzed:
        col_left, col_right = st.columns([1, 1.2])
    else:
        col_left = st.container()
        col_right = None

    with col_left:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-card-title">🩺 Describe Your Skin Symptoms</div>', unsafe_allow_html=True)

        col_age, col_dur, col_loc = st.columns(3)
        with col_age:
            age = st.number_input("Age", min_value=1, max_value=120, value=25)
        with col_dur:
            duration = st.selectbox(
                "Duration",
                ["Less than 1 week", "1-4 weeks", "1-6 months", "More than 6 months"]
            )
        with col_loc:
            location = st.selectbox(
                "Affected Body Part",
                ["Face", "Neck", "Arms", "Hands", "Chest", "Back", "Legs", "Feet", "Other"]
            )

        symptoms = st.text_area(
            "Detail observations here",
            placeholder="Example: Dark mole on arm, increasing in size, occasional itching and bleeding.",
            key="symptoms_desc",
            height=140
        )

        st.markdown('<div class="main-action-btn" style="margin-top: 20px;">', unsafe_allow_html=True)
        if st.button("Analyze Symptoms", key="symptoms_analyze_btn"):
            if "session_timestamp" in st.session_state:
                del st.session_state.session_timestamp
            symptoms_lower = symptoms.lower()
            prediction = "Unable to Determine"
            description = "Could not find explicit markers. Please provide more detail or speak with the chatbot."
            precautions = "• Keep the area clean\n• Avoid direct friction or itching\n• Consult a clinical dermatologist"

            if "mole" in symptoms_lower or "dark" in symptoms_lower or "bleeding" in symptoms_lower:
                prediction = "Melanoma"
                description = "A suspicious mole or dark lesion that changes in size, shape, or color may indicate melanoma."
                precautions = "• Consult a dermatologist immediately\n• Avoid excessive sun exposure\n• Monitor any changes carefully"
            elif "red" in symptoms_lower or "scaly" in symptoms_lower or "rough" in symptoms_lower:
                prediction = "Actinic Keratoses"
                description = "Rough, scaly patches commonly caused by long-term sun exposure (pre-cancerous)."
                precautions = "• Use sunscreen daily\n• Avoid prolonged sun exposure\n• Seek medical evaluation"
            elif "itching" in symptoms_lower or "dry" in symptoms_lower:
                prediction = "Benign Keratosis"
                description = "A common non-cancerous skin growth that may cause itching or dry skin."
                precautions = "• Keep skin moisturized\n• Avoid scratching\n• Consult a dermatologist if symptoms worsen"

            # Calculate symptom match confidence based on keyword matches
            confidence = 50.0
            if prediction == "Melanoma":
                matches = sum(1 for kw in ["mole", "dark", "bleeding"] if kw in symptoms_lower)
                confidence = 65.0 + (matches * 10.0)
            elif prediction == "Actinic Keratoses":
                matches = sum(1 for kw in ["red", "scaly", "rough"] if kw in symptoms_lower)
                confidence = 60.0 + (matches * 10.0)
            elif prediction == "Benign Keratosis":
                matches = sum(1 for kw in ["itching", "dry"] if kw in symptoms_lower)
                confidence = 60.0 + (matches * 15.0)
            else:
                confidence = 30.0

            st.session_state.symptoms_analyzed = True
            st.session_state.symptoms_prediction = prediction
            st.session_state.symptoms_description = description
            st.session_state.symptoms_precautions = precautions
            st.session_state.symptoms_age = age
            st.session_state.symptoms_duration = duration
            st.session_state.symptoms_location = location
            st.session_state.symptoms_confidence = confidence
            
            # Initialize assistant's initial message and auto-save consultation to history
            st.session_state.symptoms_messages = [
                {
                    "role": "assistant",
                    "content": (
                        f"👋 Based on your symptoms, there is a potential match for **{prediction}**.\n\n"
                        f"To help clarify this condition, could you answer a few quick questions?\n"
                        f"1. **How long have you noticed these symptoms (e.g. days, weeks, months)?**\n"
                        f"2. **Is the affected area itchy, scaly, red, or painful?**\n"
                        f"3. **Have you been exposed to prolonged sunlight or new skincare products recently?**"
                    )
                }
            ]
            save_chat_history(prediction, st.session_state.symptoms_messages)
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if is_analyzed and col_right is not None:
        with col_right:
            pred = st.session_state.symptoms_prediction
            desc = st.session_state.symptoms_description
            prec = st.session_state.symptoms_precautions

            # Color risk maps
            symptom_risk = "success"
            if pred == "Melanoma":
                symptom_risk = "danger"
            elif pred in ["Actinic Keratoses", "Unable to Determine"]:
                symptom_risk = "warning"

            badge_class = f"disease-badge disease-badge-{symptom_risk}"
            badge_icon = "⚠️" if symptom_risk == "danger" else ("🔔" if symptom_risk == "warning" else "✅")

            st.markdown(f"""
                <div class="disease-badge-container" style="text-align: center;">
                    <div class="{badge_class}">
                        <span>{badge_icon}</span>
                        <span>Potential Match: {pred}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Retrieve symptoms confidence with a fallback
            symptoms_confidence = st.session_state.get("symptoms_confidence")
            if not symptoms_confidence:
                fallback_map = {
                    "Melanoma": 85.0,
                    "Actinic Keratoses": 75.0,
                    "Benign Keratosis": 70.0,
                    "Unable to Determine": 30.0
                }
                symptoms_confidence = fallback_map.get(pred, 75.0)

            # Symptoms Confidence progress bar
            st.markdown(f"""
                <div class="confidence-container">
                    <div class="confidence-label-row">
                        <span class="confidence-title">Symptom Match Confidence</span>
                        <span class="confidence-val confidence-val-{symptom_risk}">{symptoms_confidence:.2f}%</span>
                    </div>
                    <div class="custom-progress-bg">
                        <div class="custom-progress-bar custom-progress-bar-{symptom_risk}" style="width: {symptoms_confidence}%;"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
                <div class="info-card">
                    <div class="info-card-title">📖 Description</div>
                    <div class="info-card-content">{desc}</div>
                </div>
                <div class="info-card">
                    <div class="info-card-title">🛡️ Precautions</div>
                    <div class="info-card-content">
                        <ul style="margin: 0; padding-left: 20px; line-height: 1.8;">
                            {prec.replace('•', '<li>').replace('\n', '')}
                        </ul>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Chatbot Workspace
            with st.container(border=True):
                st.markdown('<div class="chat-workspace-marker"></div>', unsafe_allow_html=True)
                st.markdown("""
                    <div class="chat-workspace-header">
                        <span>🤖</span>
                        <span>AI Symptoms Assistant</span>
                    </div>
                """, unsafe_allow_html=True)

                # Chat history init
                # Example:
                # user : hello
                # chatbot : hello what can i do for you
                if "symptoms_messages" not in st.session_state or len(st.session_state.symptoms_messages) == 0:
                    st.session_state.symptoms_messages = [
                        {
                            "role": "assistant",
                            "content": (
                                f"👋 Based on your symptoms, there is a potential match for **{pred}**.\n\n"
                                f"To help clarify this condition, could you answer a few quick questions?\n"
                                f"1. **How long have you noticed these symptoms (e.g. days, weeks, months)?**\n"
                                f"2. **Is the affected area itchy, scaly, red, or painful?**\n"
                                f"3. **Have you been exposed to prolonged sunlight or new skincare products recently?**"
                            )
                        }
                    ]

                with st.container(height=380, border=False):
                    for message in st.session_state.symptoms_messages:
                        render_chat_message(message["role"], message["content"])
                    auto_scroll()

                # Chips
                st.markdown("<div style='margin-top: 16px; margin-bottom: 8px;'><span style='font-size: 13px; font-weight: 700; color: #64748b;'>💡 Suggested Answers:</span></div>", unsafe_allow_html=True)
                symptoms_sugs = [
                    "It developed gradually over a few months.",
                    "The area is itchy and dry.",
                    f"What treatment options exist?"
                ]

                cols_chips = st.columns(len(symptoms_sugs))
                clicked_sug = None
                for i, sug in enumerate(symptoms_sugs):
                    with cols_chips[i]:
                        st.markdown('<div class="chip-marker">', unsafe_allow_html=True)
                        if st.button(sug, key=f"sug_symptoms_{i}", use_container_width=True):
                            clicked_sug = sug
                        st.markdown('</div>', unsafe_allow_html=True)

                symptoms_chat_input = st.chat_input("Ask about the analyzed condition", key="symptoms_chat_field")

                if clicked_sug:
                    symptoms_chat_input = clicked_sug

                if symptoms_chat_input:
                    st.session_state.symptoms_messages.append({"role": "user", "content": symptoms_chat_input})
                    with st.spinner("Thinking..."):
                        response = get_chatbot_response(
                            GROQ_API_KEY,
                            pred,
                            desc,
                            symptoms_chat_input,
                            st.session_state.symptoms_messages[:-1]
                        )
                    st.session_state.symptoms_messages.append({"role": "assistant", "content": response})
                    save_chat_history(pred, st.session_state.symptoms_messages)
                    st.rerun()


# ----------------------------------------------------
# 6. RESULTS PAGE (IMAGE SCANNER ANALYSIS)
# ----------------------------------------------------
elif st.session_state.page == "result":
    render_header("Detection Result")

    col_back, _ = st.columns([1.5, 8.5])
    with col_back:
        if st.button("⬅ Dashboard", key="result_back_btn", use_container_width=True):
            if "messages" in st.session_state:
                del st.session_state.messages
            st.session_state.page = "home"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    age = st.session_state.get("scanner_age", "N/A")
    duration = st.session_state.get("scanner_duration", "N/A")
    location = st.session_state.get("scanner_location", "N/A")

    disease_key = st.session_state.disease
    display_name = display_names.get(disease_key, disease_key)
    confidence = st.session_state.confidence

    # Risk Maps
    risk_mapping = {
        "akiec": "warning",
        "bcc": "danger",
        "bkl": "success",
        "df": "success",
        "mel": "danger",
        "nv": "success",
        "vasc": "success"
    }
    risk_level = risk_mapping.get(disease_key, "success")

    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        # Image Box
        st.markdown('<div class="result-image-box">', unsafe_allow_html=True)
        if "uploaded_image" in st.session_state:
            st.image(st.session_state.uploaded_image, use_container_width=True)
        elif "uploaded_image_path" in st.session_state and os.path.exists(st.session_state.uploaded_image_path):
            st.image(st.session_state.uploaded_image_path, use_container_width=True)
        else:
            st.markdown("""
                <div style="text-align: center; padding: 20px 10px; color: #64748b;">
                    <span style="font-size: 40px; display: block; margin-bottom: 8px;">📸</span>
                    <span style="font-size: 13px; font-weight: 500;">Historical Analysis Session<br>(Original image not cached)</span>
                </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Badge
        badge_class = f"disease-badge disease-badge-{risk_level}"
        badge_icon = "⚠️" if risk_level == "danger" else ("🔔" if risk_level == "warning" else "✅")
        risk_label = "High Risk / Cancerous" if risk_level == "danger" else ("Moderate Risk" if risk_level == "warning" else "Benign / Low Risk")

        st.markdown(f"""
            <div class="disease-badge-container" style="text-align: center;">
                <div class="{badge_class}">
                    <span>{badge_icon}</span>
                    <span>{risk_label}: {display_name}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Confidence Bar
        st.markdown(f"""
            <div class="confidence-container">
                <div class="confidence-label-row">
                    <span class="confidence-title">Model Confidence Score</span>
                    <span class="confidence-val confidence-val-{risk_level}">{confidence:.2f}%</span>
                </div>
                <div class="custom-progress-bg">
                    <div class="custom-progress-bar custom-progress-bar-{risk_level}" style="width: {confidence}%;"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Patient Info Table
        st.markdown(f"""
            <div class="panel-card">
                <div class="panel-card-title">👤 Patient Parameters</div>
                <table style="width:100%; border-collapse: collapse; font-size: 14.5px; color: #475569;">
                    <tr style="border-bottom: 1px solid #f1f5f9;">
                        <td style="padding: 12px 0; font-weight: 600;">Age:</td>
                        <td style="padding: 12px 0; text-align: right;">{age} years</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #f1f5f9;">
                        <td style="padding: 12px 0; font-weight: 600;">Lesion Duration:</td>
                        <td style="padding: 12px 0; text-align: right;">{duration}</td>
                    </tr>
                    <tr>
                        <td style="padding: 12px 0; font-weight: 600;">Affected Body Part:</td>
                        <td style="padding: 12px 0; text-align: right;">{location}</td>
                    </tr>
                </table>
            </div>
        """, unsafe_allow_html=True)

    with col_right:
        # Description
        st.markdown(f"""
            <div class="info-card">
                <div class="info-card-title">📖 Clinical Description</div>
                <div class="info-card-content">{disease_info.get(disease_key, "No description available.")}</div>
            </div>
        """, unsafe_allow_html=True)

        # Precautions
        st.markdown("""
            <div class="info-card">
                <div class="info-card-title">🛡️ Recommended Precautions</div>
                <div class="info-card-content">
                    <ul style="margin: 0; padding-left: 20px; line-height: 1.8;">
                        <li><strong>Dermatology Consultation:</strong> It is highly recommended to seek professional clinical confirmation.</li>
                        <li><strong>Monitor Changes:</strong> Inspect details monthly following ABCDE rules (Asymmetry, Border, Color, Diameter, Evolving).</li>
                        <li><strong>Protect From Sun:</strong> Apply sunscreen with high SPF protection and wear protective clothing when outside.</li>
                        <li><strong>Avoid Irritation:</strong> Do not scrub, pick, or attempt home treatments on the lesion.</li>
                    </ul>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # AI Assistant Chat Workspace
        with st.container(border=True):
            st.markdown('<div class="chat-workspace-marker"></div>', unsafe_allow_html=True)
            st.markdown("""
                <div class="chat-workspace-header">
                    <span>🤖</span>
                    <span>AI Consultation Workspace</span>
                </div>
            """, unsafe_allow_html=True)

            # Chat logic initialization
            # Example:
            # user : hello
            # chatbot : hello what can i do for you
            if "messages" not in st.session_state or len(st.session_state.messages) == 0:
                st.session_state.messages = [
                    {
                        "role": "assistant",
                        "content": (
                            f"👋 I have analyzed your image and detected **{display_name}** (Confidence: {confidence:.2f}%).\n\n"
                            f"To help provide relevant context, could you share a few details about the lesion?\n"
                            f"1. **Is the spot painful, itchy, bleeding, or completely asymptomatic?**\n"
                            f"2. **Have you noticed any recent changes in its size, border shape, or color?**\n"
                            f"3. **Have you consulted a dermatologist about this spot before?**"
                        )
                    }
                ]

            with st.container(height=380, border=False):
                for message in st.session_state.messages:
                    render_chat_message(message["role"], message["content"])
                auto_scroll()

            # Suggested Questions
            st.markdown("<div style='margin-top: 16px; margin-bottom: 8px;'><span style='font-size: 13px; font-weight: 700; color: #64748b;'>💡 Suggested Questions:</span></div>", unsafe_allow_html=True)
            result_sugs = [
                "The spot is itchy but does not bleed.",
                "It hasn't changed in size recently.",
                f"What next steps should I take?"
            ]

            cols_result_chips = st.columns(len(result_sugs))
            clicked_result_sug = None
            for i, sug in enumerate(result_sugs):
                with cols_result_chips[i]:
                    st.markdown('<div class="chip-marker">', unsafe_allow_html=True)
                    if st.button(sug, key=f"sug_result_{i}", use_container_width=True):
                        clicked_result_sug = sug
                    st.markdown('</div>', unsafe_allow_html=True)

            result_chat_input = st.chat_input("Ask a question about the detected disease", key="result_chat_field")

            if clicked_result_sug:
                result_chat_input = clicked_result_sug

            if result_chat_input:
                st.session_state.messages.append({"role": "user", "content": result_chat_input})
                with st.spinner("Thinking..."):
                    response = get_chatbot_response(
                        GROQ_API_KEY,
                        display_name,
                        disease_info.get(disease_key, ""),
                        result_chat_input,
                        st.session_state.messages[:-1]
                    )
                st.session_state.messages.append({"role": "assistant", "content": response})
                save_chat_history(display_name, st.session_state.messages)
                st.rerun()


# ----------------------------------------------------
# 7. APP HISTORY PAGE
# ----------------------------------------------------
elif st.session_state.page == "history":
    render_header("Consultation History")

    col_back, _ = st.columns([1.5, 8.5])
    with col_back:
        if st.button("⬅ Dashboard", key="history_back_btn", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    history_file = "chat_history.json"
    history = []
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                history = json.load(f)
        except Exception as e:
            st.error(f"Error loading history: {e}")

    if not history:
        st.info("No consultation history found. Start a new diagnosis to see your history here!")
    else:
        # Columns layout: Left column for history list, Right column for detailed conversation view
        col_list, col_detail = st.columns([1, 1.3])

        if "selected_history_index" not in st.session_state:
            st.session_state.selected_history_index = None

        with col_list:
            st.markdown('<div class="panel-card">', unsafe_allow_html=True)
            st.markdown('<div class="panel-card-title">📜 Past Consultations</div>', unsafe_allow_html=True)

            # Search bar
            search_query = st.text_input("🔍 Search by condition", key="history_search", placeholder="Type condition...")

            # Clear all history
            if st.button("🗑️ Clear All History", key="clear_all_history_btn", use_container_width=True):
                # Delete all physical image files in history
                for entry in history:
                    img_path = entry.get("image_path")
                    if img_path and os.path.exists(img_path):
                        try:
                            os.remove(img_path)
                        except Exception:
                            pass
                with open(history_file, "w") as f:
                    json.dump([], f, indent=4)
                st.session_state.selected_history_index = None
                if "session_timestamp" in st.session_state:
                    del st.session_state.session_timestamp
                st.success("History cleared successfully!")
                st.rerun()

            st.markdown("<hr style='margin: 16px 0; border: 0; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)

            # Display items in reverse chronological order
            for idx in reversed(range(len(history))):
                item = history[idx]
                disease_name = item.get("disease", "Unknown Condition")
                timestamp = item.get("timestamp", "N/A")

                # Apply search filter
                if search_query and search_query.lower() not in disease_name.lower():
                    continue

                is_selected = (st.session_state.selected_history_index == idx)

                # Determine risk level for visual markers
                risk_color = "#10b981" # success (green)
                if any(x in disease_name for x in ["Melanoma", "Carcinoma"]):
                    risk_color = "#ef4444" # danger (red)
                elif any(x in disease_name for x in ["Keratoses", "Determine"]):
                    risk_color = "#f59e0b" # warning (yellow)

                bg_color = "#f5f3ff" if is_selected else "#ffffff"
                border_color = "#7c3aed" if is_selected else "#e2e8f0"

                col_info, col_btn = st.columns([3.5, 1.5])
                with col_info:
                    st.markdown(f"""
                        <div style="background-color: {bg_color}; border: 1px solid {border_color}; border-radius: 12px; padding: 12px; border-left: 5px solid {risk_color}; height: 100%;">
                            <div style="font-size: 11px; font-weight: 600; color: #64748b; margin-bottom: 2px;">{timestamp}</div>
                            <div style="font-size: 13.5px; font-weight: 700; color: #0f172a;">{disease_name}</div>
                        </div>
                    """, unsafe_allow_html=True)

                with col_btn:
                    st.markdown('<div style="display: flex; flex-direction: column; gap: 4px; height: 100%; justify-content: center;">', unsafe_allow_html=True)
                    if st.button("🔍 View", key=f"view_{idx}", use_container_width=True):
                        st.session_state.selected_history_index = idx
                        st.rerun()
                    if st.button("🗑️ Delete", key=f"del_{idx}", use_container_width=True):
                        # Delete the physical image file
                        old_img_path = item.get("image_path")
                        if old_img_path and os.path.exists(old_img_path):
                            try:
                                os.remove(old_img_path)
                            except Exception:
                                pass
                        history.pop(idx)
                        with open(history_file, "w") as f:
                            json.dump(history, f, indent=4)
                        if st.session_state.selected_history_index == idx:
                            st.session_state.selected_history_index = None
                        elif st.session_state.selected_history_index is not None and st.session_state.selected_history_index > idx:
                            st.session_state.selected_history_index -= 1
                        st.success("Deleted!")
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

                st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        with col_detail:
            selected_idx = st.session_state.selected_history_index
            if selected_idx is not None and selected_idx < len(history):
                selected_item = history[selected_idx]
                disease_name = selected_item.get("disease", "Unknown Condition")
                timestamp = selected_item.get("timestamp", "N/A")
                messages = selected_item.get("messages", [])
                image_path = selected_item.get("image_path")

                st.markdown('<div class="panel-card">', unsafe_allow_html=True)
                st.markdown(f"""
                    <div style="border-bottom: 1px solid #f1f5f9; padding-bottom: 12px; margin-bottom: 16px;">
                        <span style="font-size: 13px; color: #64748b; font-weight: 600;">📅 Date: {timestamp}</span>
                        <h3 style="margin: 6px 0 0 0; font-size: 22px; color: #0f172a;">{disease_name}</h3>
                    </div>
                """, unsafe_allow_html=True)

                # Centered rendering for stored images if available
                if image_path and os.path.exists(image_path):
                    col_img_left, col_img_center, col_img_right = st.columns([1, 2, 1])
                    with col_img_center:
                        st.markdown('<div class="result-image-box" style="margin-bottom: 16px;">', unsafe_allow_html=True)
                        st.image(image_path, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                # Operations column
                col_act1, col_act2 = st.columns(2)
                with col_act1:
                    if st.button("🔄 Resume Consultation", key="resume_consultation_btn", use_container_width=True):
                        # Restore conversation session timestamp
                        st.session_state.session_timestamp = timestamp

                        if "General" in disease_name:
                            st.session_state.home_messages = list(messages)
                            st.session_state.chat_step = 6
                            if "uploaded_image" in st.session_state:
                                del st.session_state.uploaded_image
                            if "uploaded_image_path" in st.session_state:
                                del st.session_state.uploaded_image_path
                            st.session_state.page = "general_chatbot"
                        else:
                            # Check whether it is a scanner or symptoms checker
                            is_scanner = False
                            if messages:
                                first_msg = messages[0].get("content", "")
                                if "analyzed your image" in first_msg:
                                    is_scanner = True

                            inv_display_names = {v: k for k, v in display_names.items()}
                            disease_key = inv_display_names.get(disease_name, "akiec")

                            if is_scanner:
                                st.session_state.messages = list(messages)
                                st.session_state.disease = disease_key
                                st.session_state.confidence = 100.0  # fallback indicator
                                # Load the saved image path back into session state!
                                if image_path and os.path.exists(image_path):
                                    st.session_state.uploaded_image_path = image_path
                                    if "uploaded_image" in st.session_state:
                                        del st.session_state.uploaded_image
                                else:
                                    if "uploaded_image" in st.session_state:
                                        del st.session_state.uploaded_image
                                    if "uploaded_image_path" in st.session_state:
                                        del st.session_state.uploaded_image_path
                                    
                                st.session_state.page = "result"
                            else:
                                st.session_state.symptoms_messages = list(messages)
                                st.session_state.symptoms_prediction = disease_name
                                st.session_state.symptoms_description = disease_info.get(disease_key, "No description available.")
                                st.session_state.symptoms_precautions = "• Keep the area clean\n• Avoid direct friction or itching\n• Consult a clinical dermatologist"
                                st.session_state.symptoms_analyzed = True
                                if "uploaded_image" in st.session_state:
                                    del st.session_state.uploaded_image
                                if "uploaded_image_path" in st.session_state:
                                    del st.session_state.uploaded_image_path
                                st.session_state.page = "symptoms"

                        st.success("Resumed successfully!")
                        st.rerun()

                with col_act2:
                    # Construct download content
                    transcript_text = f"SkinScout Diagnostic Consultation Transcript\nCondition Assessed: {disease_name}\nDate/Time: {timestamp}\n"
                    transcript_text += "=" * 50 + "\n\n"
                    for msg in messages:
                        role_label = "SkinScout AI" if msg.get("role") == "assistant" else "User"
                        transcript_text += f"[{role_label}]:\n{msg.get('content')}\n\n"

                    st.download_button(
                        label="📥 Download Transcript",
                        data=transcript_text,
                        file_name=f"skinscout_consultation_{timestamp.replace(':', '-').replace(' ', '_')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

                with st.container(border=True):
                    st.markdown('<div class="chat-workspace-marker"></div>', unsafe_allow_html=True)
                    with st.container(height=380, border=False):
                        for msg in messages:
                            render_chat_message(msg.get("role", "user"), msg.get("content", ""))
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="panel-card" style="text-align: center; padding: 60px 20px;">', unsafe_allow_html=True)
                st.markdown("""
                    <span style="font-size: 48px; display: block; margin-bottom: 16px;">📜</span>
                    <h4 style="margin: 0; color: #64748b;">Select a past consultation to view the full details and transcript.</h4>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
