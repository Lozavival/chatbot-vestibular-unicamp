# Chatbot Vestibular Unicamp 2024

RAG-based chatbot that answers questions about the 2024 Unicamp admission test. Implemented as part of the selection process for undergraduate research at FEEC-NeuralMind. Implemented using [LangChain](https://www.langchain.com/) and [Groq](https://groq.com/).

## Usage Guide

After installing all the required dependencies, first run the following command to fetch the documents from the web and create the embeddings that will be fed into the model to generate the answears:

```bash
fab CreateEmbeddings
```

After the process is complete, you can run the Chatbot either via CLI or web UI, through the following commands, respectively:

```bash
fab RunChatbotCLI
```

```bash
fab RunChatbotUI
```

Alternatively, you can also run the UI Chatbot via `streamlit run src/chatbot_app.py`.
