# chatbot/chatbot.py

from groq import Groq

def get_chatbot_response(api_key, disease, description, question):
    client = Groq(api_key=api_key)

    prompt = f"""
    Detected disease: {disease}

    Description:
    {description}

    User question:
    {question}

    Give educational information and precautions.
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content