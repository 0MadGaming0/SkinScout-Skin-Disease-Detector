# chatbot/chatbot.py

from groq import Groq

# Example of single line conversation:
# user : hello
# chatbot : hello what can i do for you
def get_chatbot_response(api_key, disease, description, question, history=None):
    disclaimer = "\n\n⚠️ This is an AI-generated response based on image analysis and should not be considered a medical diagnosis. Please consult a qualified healthcare professional for accurate diagnosis and treatment."

    # Quick bypass for standard greetings to respond instantly and reliably
    clean_question = question.strip().lower().rstrip("!?. ")
    if clean_question in ["hello", "hi", "hey", "hello there", "greetings"]:
        return "Hello, what can I do for you?" + disclaimer

    client = Groq(api_key=api_key)

    system_prompt = f"""
    You are SkinScout AI, an advanced, empathetic, and professional clinical skin health assistant.
    
    Context for this session:
    - Discussed/Detected Condition: {disease}
    - Clinical Description: {description}

    Your goals and response standards:
    1. Greeting Rule: If the user provides a simple greeting (e.g., "hello", "hi", "hey"), respond with a friendly, brief welcome and ask how you can help. Do NOT preemptively detail the condition, precautions, or give clinical details until they ask.
    2. Dermatological Expertise: Provide accurate, structured, and informative answers regarding skin health, conditions, typical symptoms, general home-care precautions, and standard clinical treatments.
    3. Response Formatting: Organize information logically using clear bullet points and bold headers (such as **Overview**, **Common Symptoms**, **Precautionary Measures**, and **When to Consult a Dermatologist**).
    4. Safety & Clinical Boundaries:
       - Never prescribe specific prescription medications (e.g., specific steroids, antibiotics) or make definitive diagnoses.
       - Focus on educational descriptions, general home-care adjustments, and advice to consult healthcare practitioners.
       - Highlight key clinical "red flags" (e.g., rapid color change, irregular borders, bleeding, pain, rapid growth) if discussing potentially serious lesions.
    5. Tone: Maintain a highly professional, clinical, yet reassuring and clear tone. Avoid medical jargon overload; explain terms simply.
    6. Style & Length: Keep replies clear, relevant, and concise. Do not append any standard medical disclaimers in your text, as a standard system disclaimer is automatically added at the end of your outputs.
    """

    api_messages = [{"role": "system", "content": system_prompt}]

    if history:
        for msg in history:
            role = msg.get("role")
            content = msg.get("content", "")
            
            # Clean up disclaimer from assistant's past messages
            if role == "assistant" and disclaimer in content:
                content = content.replace(disclaimer, "").strip()
                
            api_messages.append({"role": role, "content": content})

    api_messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=api_messages
    )

    raw_response = response.choices[0].message.content

    # Clean up common LLM-generated disclaimers to prevent duplicates
    if any(phrase in raw_response.lower() for phrase in ["medical diagnosis", "ai-generated response", "qualified healthcare professional"]):
        lines = raw_response.splitlines()
        filtered_lines = [
            line for line in lines 
            if not any(phrase in line.lower() for phrase in ["medical diagnosis", "ai-generated response", "qualified healthcare professional", "should not be considered"])
        ]
        raw_response = "\n".join(filtered_lines).strip()

    return raw_response + disclaimer