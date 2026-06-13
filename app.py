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

GROQ_API_KEY = "YOUR_GROQ_API_KEY"
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
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            history = json.load(f)
    else:
        history = []
    
    history.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "disease": disease,
        "messages": messages
    })
    
    with open(history_file, "w") as f:
        json.dump(history, f, indent=4)

# Global Navbar Component
def render_header(page_title):
    if st.session_state.get("logged_in") and st.session_state.page != "login":
        col_title, col_user, col_logout = st.columns([7, 2.3, 1.7])
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
                    <span class="nav-user-badge" style="margin-top: 4px;">👤 {st.session_state.username}</span>
                </div>
            """, unsafe_allow_html=True)
        with col_logout:
            st.markdown('<div class="nav-logout-marker" style="display: flex; align-items: center; height: 100%;">', unsafe_allow_html=True)
            if st.button("🚪 Logout", key="global_logout_btn"):
                st.session_state.logged_in = False
                st.session_state.page = "login"
                st.query_params.clear()
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
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

    st.markdown("""
        <div style="margin-bottom: 40px; text-align: center;">
            <h2 style="font-weight: 800; font-size: 28px; color: #0f172a; margin-bottom: 10px; letter-spacing: -0.5px;">Intelligent Skin Care Insights</h2>
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
            st.rerun()

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Centered glowing neon chatbot button at the bottom center
    _, center_col, _ = st.columns([1.5, 1.4, 1.5])
    with center_col:
        st.markdown('<div class="chatbot-btn-marker"></div>', unsafe_allow_html=True)
        if st.button("💬 AI Assistant Chatbot", key="home_chatbot_btn", use_container_width=True):
            st.session_state.page = "general_chatbot"
            st.rerun()


# ----------------------------------------------------
# 3. AI GENERAL CHATBOT PAGE
# ----------------------------------------------------
elif st.session_state.page == "general_chatbot":
    render_header("AI Assessment Assistant")

    col_back, _ = st.columns([1.5, 8.5])
    with col_back:
        if st.button("⬅ Dashboard", key="chatbot_back_btn", use_container_width=True):
            st.session_state.page = "home"
            if "chat_step" in st.session_state:
                del st.session_state.chat_step
            if "chat_answers" in st.session_state:
                del st.session_state.chat_answers
            st.session_state.home_messages = []
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Initialize guided interview states
    if "home_messages" not in st.session_state or len(st.session_state.home_messages) == 0:
        st.session_state.home_messages = [
            {
                "role": "assistant",
                "content": "👋 Hello! I am your SkinScout Assistant. Let's do a quick skin assessment.\n\n**Question 1:** Are you having any type of skin symptoms (like a rash, mole, dry skin, or lesion)?"
            }
        ]
        st.session_state.chat_step = 1
        st.session_state.chat_answers = {}

    current_step = st.session_state.get("chat_step", 1)

    # Render steps dot container if still within assessment
    if current_step <= 5:
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
                <span class="chat-progress-label">📋 GUIDED ASSESSMENT: {step_labels[current_step]}</span>
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
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    user_question = st.chat_input("Type your response or question here...", key="general_chat_input")

    if user_question:
        # Append user message
        st.session_state.home_messages.append({"role": "user", "content": user_question})

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
            st.rerun()

        else:
            # Free-form follow-up conversation after assessment
            with st.spinner("Thinking..."):
                response = get_chatbot_response(
                    GROQ_API_KEY,
                    "General Skin Health (Follow-up)",
                    "Follow-up discussion after guided skin assessment.",
                    user_question
                )
            st.session_state.home_messages.append({"role": "assistant", "content": response})
            st.rerun()

    # Clear/Reset Chat trigger
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑 Reset Assessment"):
        st.session_state.home_messages = []
        if "chat_step" in st.session_state:
            del st.session_state.chat_step
        if "chat_answers" in st.session_state:
            del st.session_state.chat_answers
        st.rerun()


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
            image_source = uploaded_file if uploaded_file else camera_image
            image = Image.open(image_source).convert("RGB")
            image = transform(image).unsqueeze(0)

            with st.spinner("Processing image and running predictions..."):
                with torch.no_grad():
                    outputs = model(image)
                    probabilities = torch.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)

            st.session_state.disease = classes[predicted.item()]
            st.session_state.confidence = confidence.item() * 100
            st.session_state.uploaded_image = image_source
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

            st.session_state.symptoms_analyzed = True
            st.session_state.symptoms_prediction = prediction
            st.session_state.symptoms_description = description
            st.session_state.symptoms_precautions = precautions
            st.session_state.symptoms_age = age
            st.session_state.symptoms_duration = duration
            st.session_state.symptoms_location = location
            st.session_state.symptoms_messages = []
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
            st.markdown("""
                <div class="chat-workspace">
                    <div class="chat-workspace-header">
                        <span>🤖</span>
                        <span>AI Symptoms Assistant</span>
                    </div>
            """, unsafe_allow_html=True)

            # Chat history init
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

            for message in st.session_state.symptoms_messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

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
                        symptoms_chat_input
                    )
                st.session_state.symptoms_messages.append({"role": "assistant", "content": response})
                save_chat_history(pred, st.session_state.symptoms_messages)
                st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)


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
                    <span class="confidence-val">{confidence:.2f}%</span>
                </div>
                <div class="custom-progress-bg">
                    <div class="custom-progress-bar" style="width: {confidence}%;"></div>
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
        st.markdown("""
            <div class="chat-workspace">
                <div class="chat-workspace-header">
                    <span>🤖</span>
                    <span>AI Consultation Workspace</span>
                </div>
        """, unsafe_allow_html=True)

        # Chat logic initialization
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

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

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
                    result_chat_input
                )
            st.session_state.messages.append({"role": "assistant", "content": response})
            save_chat_history(display_name, st.session_state.messages)
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
