import streamlit as st
import torch
import pickle
from PIL import Image
from torchvision import transforms
from groq import Groq
from chatbot.chatbot import get_chatbot_response

import os

client = Groq(api_key=  os.getenv("GROQ_API_KEY"))

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

# Session State
if "page" not in st.session_state:
    st.session_state.page = "login"

# LOGIN PAGE
if st.session_state.page == "login":
    st.title("🏥 AI-Powered Skin Disease Detection")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state.page = "home"
            st.rerun()

        else:
            st.error("Invalid Login")

# HOME PAGE
elif st.session_state.page == "home":

    st.title("🏠 Home")
    st.write("Upload a skin image or use your camera.")

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

    st.subheader("🩺 Symptoms Information")

    age = st.number_input(
           "Age",
        min_value=0,
        max_value=120,
        value=25
    )

    gender = st.selectbox(
    "Gender",
    ["Male", "Female", "Other"]
   )

    symptoms = st.text_area(
    "Describe your symptoms",
    placeholder="Example: Itching, bleeding, pain, change in color..."
    )

    duration = st.text_input(
    "How long have you noticed it?",
    placeholder="Example: 2 weeks"
    )
    


    if st.button("Analyze Symptoms"):

      prompt = f"""
      Age: {age}
      Gender: {gender}

      Symptoms:
      {symptoms}

      Duration:
      {duration}

      Give:
      1. Possible skin conditions
      2. General precautions
      3. Whether medical consultation is recommended

      Do not provide a definitive diagnosis.
      """

      response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
     ]
      )

      st.subheader("🤖 Symptom Analysis")
      st.write(response.choices[0].message.content)

    if st.button("Detect Disease"):

        if uploaded_file or camera_image:

            image_source = uploaded_file if uploaded_file else camera_image

            st.session_state.uploaded_image = image_source

            image = Image.open(image_source).convert("RGB")
            image = transform(image).unsqueeze(0)

            with torch.no_grad():
                outputs = model(image)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

            st.session_state.disease = classes[predicted.item()]
            print("Predicted:", classes[predicted.item()])
            print("Confidence:", confidence.item() * 100)
            st.session_state.confidence = confidence.item() * 100

            st.session_state.page = "result"
            st.rerun()

        else:
            st.warning("Please upload or capture an image first.")
# RESULT PAGE
elif st.session_state.page == "result":

    st.title("📊 Detection Result")

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

    user_question = st.text_input(
    "Ask a question about the detected disease"
     )

    if user_question:

      disease = display_names[st.session_state.disease]

      response = get_chatbot_response(
        GROQ_API_KEY,
        disease,
        disease_info[st.session_state.disease],
        user_question
    )

      st.write(response)

    if st.button("Back To Home"):
        st.session_state.page = "home"
        st.rerun()