# ---------- CHATBOT ----------
import os
from dotenv import load_dotenv
from groq import Groq


if load_dotenv(".venv/.env"):
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
else:
    raise Exception("Failed to load env variables")


def chatbot(user_query):
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": user_query,
            }
        ],
        model="llama3-70b-8192",
    )
    return response.choices[0].message.content


# ---------- UI ----------
import streamlit as st
from streamlit_chat import message

# Streamlit UI setup
st.title("RAG-based Chatbot")

# Initialize session state for messages
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Display messages
for message_dict in st.session_state['messages']:
    message(**message_dict)

with st.form(key='chat_form', clear_on_submit=True):
    user_input = st.text_area("Enter a question:", value="", key='input')
    submit_button = st.form_submit_button(label="Enviar")

if submit_button and user_input:
    # Append user query to session messages
    st.session_state['messages'].append({"message": user_input, "is_user": True})
    # Get chatbot response
    response = chatbot(user_input)
    # Append chatbot response to session messages
    st.session_state['messages'].append({"message": response, "is_user": False})
    # Rerun the app to update the UI
    st.rerun()
