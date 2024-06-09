import streamlit as st
from streamlit_chat import message

from chatbot import chatbot

st.title("Chatbot Vestibular Unicamp 2024")

# Initialize session state for messages
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Display messages
for message_dict in st.session_state['messages']:
    message(**message_dict)

with st.form(key='chat_form', clear_on_submit=True):
    user_input    = st.text_area("Faça uma pergunta:", value="", key='input')
    submit_button = st.form_submit_button(label="Enviar")

if submit_button and user_input:
    st.session_state['messages'].append({"message": user_input, "is_user": True})
    response = chatbot(user_input)
    st.session_state['messages'].append({"message": response, "is_user": False})

    st.rerun()
