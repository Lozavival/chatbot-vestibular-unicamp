import os

import bs4
from fabric import task

from src import chatbot


@task
def CreateEmbeddings(c):
    url      = "https://www.pg.unicamp.br/norma/31594/0"
    strainer = bs4.SoupStrainer(class_=("card-body"))
    docs     = chatbot.get_documents(url, strainer)
    chatbot.create_embeddings_vectorstore(docs)


@task
def RunChatbotCLI(c):
    rag_chain = chatbot.create_rag_chain()
    while (user_input := input("Pergunta: ")):
        print(chatbot.chatbot(user_input, rag_chain))


@task
def RunChatbotUI(c):
    os.system("streamlit run src/chatbot_app.py")
