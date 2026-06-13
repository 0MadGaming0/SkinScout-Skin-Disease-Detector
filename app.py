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

GROQ_API_KEY = "+dYOUR_GROQ_API_KEY_HERE"

client = Groq(api_key=GROQ_API_KEY)

st.set_page_config(page_title="Skin Disease Detection", layout="wide")
# Load model

original_load = torch.load

def cpu_load(*args, **kwargs):
    kwargs["map_location"] = torch.device("cpu")
    return original_load(*args, **kwargs)

torch.load = cpu_load

with open("skin_disease_model.pkl", "rb") as f:
    model = pickle.load(f)

model.eval()


print("Model loaded successfully")


classes = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]
disease_info = {
    "akiec": "A rough, scaly patch on the skin caused by years of sun exposure.",
    "bcc": "A common type of skin cancer that usually develops on sun-exposed skin.",
    "bkl": "A harmless skin growth that often appears with aging.",
    "df": "A benign skin nodule that usually appears on the legs or arms.",
    "mel": "A serious form of skin cancer that can spread if untreated.",
    "nv": "Common benign moles formed by pigment-producing cells.",
    "vasc": "Skin abnormalities involving blood vessels, such as angiomas."
}

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
# Save chat history function
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

# Session State
if "logged_in" not in st.session_state:
    if st.query_params.get("logged_in") == "true":
        st.session_state.logged_in = True
        st.session_state.username = st.query_params.get("username", "admin")
        st.session_state.page = "home"
    else:
        st.session_state.logged_in = False

if "page" not in st.session_state:
    st.session_state.page = "login"
    
if (
    st.session_state.logged_in
    and st.session_state.page == "login"
):
    st.session_state.page = "home"
# LOGIN PAGE
if st.session_state.page == "login":
    # Inject styling
    st.markdown("""
        <style>
        /* Background styling - soft sky blue gradients */
        div[data-testid="stAppViewContainer"]:has(.login-card-marker) {
            background: radial-gradient(circle at 15% 15%, rgba(14, 165, 233, 0.08) 0%, transparent 45%),
                        radial-gradient(circle at 85% 85%, rgba(99, 102, 241, 0.08) 0%, transparent 45%),
                        #f8fafc !important;
        }

        /* Hide standard Streamlit header on the login page */
        div[data-testid="stAppViewContainer"]:has(.login-card-marker) header {
            display: none !important;
        }

        /* Style the column containing the marker as the login card */
        div[data-testid="stColumn"]:has(.login-card-marker) {
            background: #ffffff !important;
            border: 1px solid #f1f5f9 !important;
            border-radius: 24px !important;
            padding: 3.5rem 2.5rem !important;
            box-shadow: 0 25px 50px -12px rgba(15, 23, 42, 0.08) !important;
            margin-top: 4rem !important;
        }

        /* Style the text inputs inside the login card */
        div[data-testid="stColumn"]:has(.login-card-marker) div[data-testid="stTextInput"] input {
            background-color: #f8fafc !important;
            color: #0f172a !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 12px !important;
            padding: 12px 16px !important;
            font-size: 15px !important;
            transition: all 0.2s ease-in-out !important;
            box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.02) !important;
        }

        div[data-testid="stColumn"]:has(.login-card-marker) div[data-testid="stTextInput"] input:focus {
            border-color: #0ea5e9 !important;
            background-color: #ffffff !important;
            box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.15) !important;
        }

        /* Style input labels */
        div[data-testid="stColumn"]:has(.login-card-marker) div[data-testid="stTextInput"] label {
            color: #475569 !important;
            font-weight: 600 !important;
            font-size: 14px !important;
            margin-bottom: 8px !important;
            letter-spacing: 0.3px !important;
        }

        /* Style the login button inside the login card */
        div[data-testid="stColumn"]:has(.login-card-marker) div[data-testid="stButton"] button {
            background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 12px 24px !important;
            font-size: 16px !important;
            font-weight: 600 !important;
            width: 100% !important;
            margin-top: 1rem !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-shadow: 0 4px 12px rgba(14, 165, 233, 0.25) !important;
        }

        div[data-testid="stColumn"]:has(.login-card-marker) div[data-testid="stButton"] button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 20px rgba(14, 165, 233, 0.35) !important;
            background: linear-gradient(135deg, #38bdf8 0%, #0ea5e9 100%) !important;
        }

        div[data-testid="stColumn"]:has(.login-card-marker) div[data-testid="stButton"] button:active {
            transform: translateY(0) !important;
        }

        /* Custom login header styling */
        .login-header-container {
            text-align: center;
            margin-bottom: 2rem;
        }

        .login-logo {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
            color: #0284c7;
            width: 64px;
            height: 64px;
            border-radius: 20px;
            font-size: 32px;
            margin-bottom: 1rem;
            box-shadow: 0 8px 16px -4px rgba(14, 165, 233, 0.2);
        }

        .login-title {
            color: #0f172a;
            font-size: 28px;
            font-weight: 800;
            margin: 0;
            letter-spacing: -0.5px;
            line-height: 1.2;
        }

        .login-subtitle {
            color: #64748b;
            font-size: 15px;
            margin-top: 0.5rem;
            margin-bottom: 0;
        }

        </style>
    """, unsafe_allow_html=True)

    # Centered layout using 3 columns
    _, col, _ = st.columns([1, 1.8, 1])

    with col:
        # Marker element to target this column specifically
        st.markdown('<div class="login-card-marker"></div>', unsafe_allow_html=True)

        # Stylized Header
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


# HOME PAGE
elif st.session_state.page == "home":

    st.title("🏥 SkinScout")

    st.info(
        f"👤 Logged in as: {st.session_state.username}"
    )
    
    if st.button("🚪 Logout"):

        st.session_state.logged_in = False
        st.session_state.page = "login"
        st.query_params.clear()

        st.rerun()

    st.write(
        "Welcome to SkinScout - AI-Powered Skin Disease Detection and Assistance."
    )

    st.markdown("""
        <style>
        /* Stylized Home Page Chatbot Button (Button 64 Theme) */
        div[data-testid="stColumn"]:has(.chatbot-btn-marker) button {
            background-image: linear-gradient(#090d16, #090d16), linear-gradient(135deg, #8b5cf6 0%, #0ea5e9 100%) !important;
            background-origin: border-box !important;
            background-clip: padding-box, border-box !important;
            border: 2px solid transparent !important;
            color: #ffffff !important;
            font-size: 18px !important;
            font-weight: 600 !important;
            border-radius: 12px !important;
            padding: 14px 28px !important;
            box-shadow: 0 10px 30px rgba(139, 92, 246, 0.25), 0 0 15px rgba(14, 165, 233, 0.25) !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            letter-spacing: 0.5px !important;
            text-align: center !important;
            width: 100% !important;
        }
        
        div[data-testid="stColumn"]:has(.chatbot-btn-marker) button:hover {
            transform: translateY(-2px) scale(1.02) !important;
            box-shadow: 0 15px 35px rgba(139, 92, 246, 0.4), 0 0 20px rgba(14, 165, 233, 0.4) !important;
            background-image: linear-gradient(#0c1220, #0c1220), linear-gradient(135deg, #a78bfa 0%, #38bdf8 100%) !important;
        }
        
        div[data-testid="stColumn"]:has(.chatbot-btn-marker) button:active {
            transform: translateY(0) scale(1.0) !important;
        }
        </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📸 Image Diagnosis")

        st.write(
            "Upload or capture a skin image for AI-based disease detection."
        )

        if st.button("Open Scanner"):
            st.session_state.page = "scanner"
            st.rerun()

    with col2:

        st.subheader("🩺 Symptoms Checker")

        st.write(
            "Analyze symptoms without uploading an image."
        )

        if st.button("Open Symptoms Checker"):
            st.session_state.page = "symptoms"
            st.rerun()

    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Bottom Center columns for perfect centering
    _, center_col, _ = st.columns([1.5, 1, 1.5])
    with center_col:
        st.markdown('<div class="chatbot-btn-marker"></div>', unsafe_allow_html=True)
        if st.button("Chatbot", key="home_chatbot_btn", use_container_width=True):
            st.session_state.page = "general_chatbot"
            st.rerun()


# CHATBOT PAGE
elif st.session_state.page == "general_chatbot":

    st.title("🤖 SkinScout Assistant")

    if st.button("⬅ Back To Home"):
        st.session_state.page = "home"
        if "chat_step" in st.session_state:
            del st.session_state.chat_step
        if "chat_answers" in st.session_state:
            del st.session_state.chat_answers
        st.session_state.home_messages = []
        st.rerun()

    st.markdown("---")

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

    # Display previous messages
    for message in st.session_state.home_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    user_question = st.chat_input(
        "Type your answer here...",
        key="general_chat_input"
    )

    if user_question:
        # Append user's answer
        st.session_state.home_messages.append(
            {
                "role": "user",
                "content": user_question
            }
        )

        current_step = st.session_state.get("chat_step", 1)

        if current_step < 5:
            # Save the answer for the current question
            st.session_state.chat_answers[current_step] = user_question
            next_step = current_step + 1
            st.session_state.chat_step = next_step

            # Define next questions
            questions = {
                2: "**Question 2:** How long have you noticed these symptoms (e.g., days, weeks, months)?",
                3: "**Question 3:** Where on your body is the affected area located?",
                4: "**Question 4:** Is the area painful, itchy, bleeding, or scaling?",
                5: "**Question 5:** Have you had a history of skin allergies or similar conditions?"
            }

            # Ask next question
            st.session_state.home_messages.append(
                {
                    "role": "assistant",
                    "content": questions[next_step]
                }
            )
            st.rerun()

        elif current_step == 5:
            # Save final answer
            st.session_state.chat_answers[5] = user_question
            st.session_state.chat_step = 6

            # Call API to analyze responses
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

                st.session_state.home_messages.append(
                    {
                        "role": "assistant",
                        "content": f"📋 **Assessment Analysis Result**:\n\n{response}\n\n*Feel free to ask me any follow-up questions below.*"
                    }
                )
            st.rerun()

        else:
            # Free-form follow-up conversation after step 5
            response = get_chatbot_response(
                GROQ_API_KEY,
                "General Skin Health (Follow-up)",
                "Follow-up discussion after guided skin assessment.",
                user_question
            )

            st.session_state.home_messages.append(
                {
                    "role": "assistant",
                    "content": response
                }
            )
            st.rerun()

    # Clear Chat button
    if st.button("🗑 Reset Assessment"):
        st.session_state.home_messages = []
        if "chat_step" in st.session_state:
            del st.session_state.chat_step
        if "chat_answers" in st.session_state:
            del st.session_state.chat_answers
        st.rerun()
# SCANNER PAGE
elif st.session_state.page == "scanner":

    st.title("📸 Image Diagnosis")

    if st.button("⬅ Back"):
        st.session_state.page = "home"
        st.rerun()

    input_method = st.radio(
        "Choose Input Method",
        ["Upload Image", "Capture Image"]
    )

    if input_method == "Upload Image":
        uploaded_file = st.file_uploader(
            "Upload Skin Image",
            type=["jpg", "jpeg", "png"]
        )
        camera_image = None

    else:
        camera_image = st.camera_input("Capture Image")
        uploaded_file = None

    if uploaded_file:
        st.image(uploaded_file)

    if camera_image:
        st.image(camera_image)

    st.subheader("📋 Patient & Lesion Details (Optional)")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=25,
            key="scanner_age"
        )
    with col2:
        duration = st.selectbox(
            "How long have you had this lesion?",
            [
                "Less than 1 week",
                "1-4 weeks",
                "1-6 months",
                "More than 6 months"
            ],
            key="scanner_duration"
        )
    with col3:
        location = st.selectbox(
            "Affected Body Part",
            [
                "Face",
                "Neck",
                "Arms",
                "Hands",
                "Chest",
                "Back",
                "Legs",
                "Feet",
                "Other"
            ],
            key="scanner_location"
        )

    st.divider()

    if st.button("🔍 Detect Disease"):

        if uploaded_file or camera_image:

            image_source = uploaded_file if uploaded_file else camera_image

            image = Image.open(image_source).convert("RGB")
            image = transform(image).unsqueeze(0)

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
            st.warning("Please upload or capture an image first.")
# SYMPTOMS PAGE
elif st.session_state.page == "symptoms":

    st.title("🩺 Symptoms Checker")

    if st.button("⬅ Back To Home"):
        st.session_state.page = "home"
        if "symptoms_analyzed" in st.session_state:
            del st.session_state.symptoms_analyzed
        if "symptoms_messages" in st.session_state:
            del st.session_state.symptoms_messages
        st.rerun()

    st.markdown("---")

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=25
    )

    duration = st.selectbox(
        "How long have you had these symptoms?",
        [
            "Less than 1 week",
            "1-4 weeks",
            "1-6 months",
            "More than 6 months"
        ]
    )
    location = st.selectbox(
        "Affected Body Part",
        [
            "Face",
            "Neck",
            "Arms",
            "Hands",
            "Chest",
            "Back",
            "Legs",
            "Feet",
            "Other"
        ]
    )

    symptoms = st.text_area(
        "Describe Your Symptoms",
        placeholder="""
Example:
Dark mole on arm, increasing in size,
occasional itching and bleeding.
"""
    )

    if st.button("Analyze Symptoms"):

        symptoms_lower = symptoms.lower()

        prediction = "Unable to Determine"

        description = ""
        precautions = ""

        if (
            "mole" in symptoms_lower
            or "dark" in symptoms_lower
            or "bleeding" in symptoms_lower
        ):

            prediction = "Melanoma"

            description = """
A suspicious mole or dark lesion that changes
in size, shape, or color may indicate melanoma.
"""

            precautions = """
• Consult a dermatologist immediately

• Avoid excessive sun exposure

• Monitor any changes carefully
"""

        elif (
            "red" in symptoms_lower
            or "scaly" in symptoms_lower
            or "rough" in symptoms_lower
        ):

            prediction = "Actinic Keratoses"

            description = """
Rough, scaly patches commonly caused by long-term
sun exposure.
"""

            precautions = """
• Use sunscreen daily

• Avoid prolonged sun exposure

• Seek medical evaluation
"""

        elif (
            "itching" in symptoms_lower
            or "dry" in symptoms_lower
        ):

            prediction = "Benign Keratosis"

            description = """
A common non-cancerous skin condition that may
cause itching or dry skin.
"""

            precautions = """
• Keep skin moisturized

• Avoid scratching

• Consult a dermatologist if symptoms worsen
"""

        st.session_state.symptoms_analyzed = True
        st.session_state.symptoms_prediction = prediction
        st.session_state.symptoms_description = description
        st.session_state.symptoms_precautions = precautions
        st.session_state.symptoms_age = age
        st.session_state.symptoms_duration = duration
        st.session_state.symptoms_location = location
        st.session_state.symptoms_messages = []

    if st.session_state.get("symptoms_analyzed"):
        st.divider()
        st.success(
            f"Possible Condition: {st.session_state.symptoms_prediction}"
        )

        st.write(f"Age: {st.session_state.symptoms_age}")
        st.write(f"Duration: {st.session_state.symptoms_duration}")
        st.write(f"Affected Area: {st.session_state.symptoms_location}")

        st.subheader("Description")
        st.write(st.session_state.symptoms_description)

        st.subheader("Precautions")
        st.write(st.session_state.symptoms_precautions)

        st.subheader("🤖 Ask About Your Symptoms")

        # Initialize messages if not present
        if "symptoms_messages" not in st.session_state or len(st.session_state.symptoms_messages) == 0:
            st.session_state.symptoms_messages = [
                {
                    "role": "assistant",
                    "content": (
                        f"👋 Based on your symptoms, there is a potential match for **{st.session_state.symptoms_prediction}**.\n\n"
                        f"To help clarify this condition, could you answer a few quick questions?\n"
                        f"1. **How long have you noticed these symptoms (e.g. days, weeks, months)?**\n"
                        f"2. **Is the affected area itchy, scaly, red, or painful?**\n"
                        f"3. **Have you been exposed to prolonged sunlight or new skincare products recently?**"
                    )
                }
            ]

        # Display chatbot history
        for message in st.session_state.symptoms_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Suggested questions (Answers to the chatbot's questions)
        symptoms_sugs = [
            "It developed gradually over a few months.",
            "The area is itchy and dry.",
            f"What are the typical treatment options for {st.session_state.symptoms_prediction}?"
        ]
        
        st.markdown("💡 **Suggested Questions:**")
        cols = st.columns(len(symptoms_sugs))
        clicked_symptoms_sug = None
        for i, sug in enumerate(symptoms_sugs):
            with cols[i]:
                if st.button(sug, key=f"sug_symptoms_{i}", use_container_width=True):
                    clicked_symptoms_sug = sug

        # Chat input
        symptoms_user_question = st.chat_input(
            "Ask a question about the analyzed condition",
            key="symptoms_chat_input"
        )

        if clicked_symptoms_sug:
            symptoms_user_question = clicked_symptoms_sug

        if symptoms_user_question:
            st.session_state.symptoms_messages.append(
                {"role": "user", "content": symptoms_user_question}
            )

            response = get_chatbot_response(
                GROQ_API_KEY,
                st.session_state.symptoms_prediction,
                st.session_state.symptoms_description,
                symptoms_user_question
            )

            st.session_state.symptoms_messages.append(
                {"role": "assistant", "content": response}
            )
            save_chat_history(
                st.session_state.symptoms_prediction,
                st.session_state.symptoms_messages
            )

            st.rerun()
# RESULT PAGE
elif st.session_state.page == "result":

    st.title("📊 Detection Result")

    age = st.session_state.get("scanner_age", "N/A")
    duration = st.session_state.get("scanner_duration", "N/A")
    location = st.session_state.get("scanner_location", "N/A")

    display_names = {
    "akiec": "Actinic Keratoses",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevus",
    "vasc": "Vascular Lesion"
}

    if "uploaded_image" in st.session_state:
        st.image(st.session_state.uploaded_image, width=300)

    st.success(
        f"Disease Detected: {display_names[st.session_state.disease]}"
    )

    st.write(
    f"Confidence Score: {st.session_state.confidence:.2f}%"
)

    st.write("Raw Model Output:")
    st.write(st.session_state.disease)
    st.write(f"Age: {age}")
    st.write(f"Duration: {duration}")
    st.write(f"Affected Area: {location}")
    
    st.subheader("Description")
    st.write(
        disease_info[st.session_state.disease]
    )

    st.subheader("Precautions")
    st.write("""
     • Consult a dermatologist

     • Follow proper skin care

     • Avoid self-medication

     • Monitor any changes in size, color, or shape
     """)
    st.subheader("🤖 Ask About Your Condition")

    # Initialize chat history
    if "messages" not in st.session_state or len(st.session_state.messages) == 0:
        disease_name = display_names[st.session_state.disease]
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    f"👋 I have analyzed your image and detected **{disease_name}** (Confidence: {st.session_state.confidence:.2f}%).\n\n"
                    f"To help provide relevant context, could you share a few details about the lesion?\n"
                    f"1. **Is the spot painful, itchy, bleeding, or completely asymptomatic?**\n"
                    f"2. **Have you noticed any recent changes in its size, border shape, or color?**\n"
                    f"3. **Have you consulted a dermatologist about this spot before?**"
                )
            }
        ]

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Suggested questions (Answers to the chatbot's questions)
    disease_name = display_names[st.session_state.disease]
    result_sugs = [
        "The spot is itchy but does not bleed.",
        "It hasn't changed in size recently.",
        f"What next steps should I take for {disease_name}?"
    ]
    
    st.markdown("💡 **Suggested Questions:**")
    cols = st.columns(len(result_sugs))
    clicked_result_sug = None
    for i, sug in enumerate(result_sugs):
        with cols[i]:
            if st.button(sug, key=f"sug_result_{i}", use_container_width=True):
                clicked_result_sug = sug

    # Chat input
    user_question = st.chat_input(
        "Ask a question about the detected disease"
    )

    if clicked_result_sug:
        user_question = clicked_result_sug

    if user_question:

        # Store user message
        st.session_state.messages.append(
            {"role": "user", "content": user_question}
        )

        disease = display_names[st.session_state.disease]

        response = get_chatbot_response(
            GROQ_API_KEY,
            disease,
            disease_info[st.session_state.disease],
            user_question
        )

        # Store bot response
        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )
        save_chat_history(
            display_names[st.session_state.disease],
            st.session_state.messages
        )

        st.rerun()

    # Back button (OUTSIDE the if user_question block)
    if st.button("Back To Home"):

        if "messages" in st.session_state:
            del st.session_state.messages

        st.session_state.page = "home"
        st.rerun()
