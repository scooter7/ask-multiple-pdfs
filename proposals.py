import os
import streamlit as st
import requests
from PyPDF2 import PdfReader
from io import BytesIO
import numpy as np
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

# Adjusted to point to the 'rfps' folder
GITHUB_REPO_URL = "https://github.com/scooter7/ask-multiple-pdfs/tree/main/rfps/"

def get_github_pdfs(repo_url):
    api_url = "https://api.github.com/repos/scooter7/ask-multiple-pdfs/contents/rfps"
    headers = {'Accept': 'application/vnd.github.v3+json'}
    response = requests.get(api_url, headers=headers)
    files = response.json()
    
    pdf_docs = []
    for file in files:
        if file['name'].endswith('.pdf'):
            pdf_url = file['download_url']
            response = requests.get(pdf_url)
            pdf_docs.append(BytesIO(response.content))
    return pdf_docs

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    if not text_chunks:
        raise ValueError("No text chunks available for embedding.")
    os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)
    return conversation_chain

def modify_response_language(original_response):
    response = original_response.replace(" they ", " we ")
    response = response.replace("They ", "We ")
    response = response.replace(" their ", " our ")
    response = response.replace("Their ", "Our ")
    return response

def handle_userinput(user_question, requirements_text, vectorstore, text_chunks):
    if 'conversation' not in st.session_state:
        st.error("The conversational model is not initialized properly.")
        return

    if requirements_text:
        requirements_embedding = vectorstore.embedding.embed([requirements_text])[0]
        scores, indices = vectorstore.faiss_index.search(np.array([requirements_embedding]), k=5)
        combined_context = " ".join([text_chunks[idx] for idx in indices[0]])
    else:
        combined_context = ""
    
    prompt = f"Context: {combined_context}\n\nQuestion: {user_question}"
    response = st.session_state.conversation({'question': prompt})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        modified_content = modify_response_language(message.content)
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", modified_content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", modified_content), unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="CAI", page_icon="https://www.carnegiehighered.com/wp-content/uploads/2021/11/Twitter-Image-2-2021.png")
    st.write(css, unsafe_allow_html=True)

    header_html = """
    <div style="text-align: center;">
        <h1 style="font-weight: bold;">Carnegie Artifical Intelligence - CAI</h1>
        <img src="https://www.carnegiehighered.com/wp-content/uploads/2021/11/Twitter-Image-2-2021.png" alt="Icon" style="height:200px; width:500px;">
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload your requirements PDF", type="pdf")
    requirements_text = ""
    if uploaded_file:
        requirements_text = get_pdf_text([uploaded_file])

    pdf_docs = get_github_pdfs(GITHUB_REPO_URL)
    if pdf_docs:
        raw_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(raw_text)
        if text_chunks:
            vectorstore = get_vectorstore(text_chunks)
            st.session_state.conversation = get_conversation_chain(vectorstore)
        else:
            st.error("No text chunks available from the 'rfps' documents.")
    else:
        st.error("No PDF documents found in the specified GitHub 'rfps' folder.")

    user_question = st.text_input("Ask CAI about anything Carnegie:")
    if user_question:
        handle_userinput(user_question, requirements_text, vectorstore, text_chunks)

if __name__ == '__main__':
    main()
