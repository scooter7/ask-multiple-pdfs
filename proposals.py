import os
import requests
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from io import BytesIO

def fetch_pdfs_from_github(github_url):
    response = requests.get(github_url)
    data = response.json()
    pdf_urls = [file['download_url'] for file in data if file['name'].endswith('.pdf')]
    return pdf_urls

def download_pdfs(pdf_urls):
    pdf_docs = []
    for url in pdf_urls:
        response = requests.get(url)
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

def initialize_conversation(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)
    return conversation_chain

def handle_userinput(conversation_chain, user_question):
    if not conversation_chain:
        st.error("The conversation model is not initialized.")
        return

    response = conversation_chain({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Proposal Exploration Tool", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # Load and process the existing PDFs from GitHub
    github_url = "https://api.github.com/repos/scooter7/ask-multiple-pdfs/contents/rfps"
    pdf_urls = fetch_pdfs_from_github(github_url)
    existing_pdfs = download_pdfs(pdf_urls)
    existing_text = get_pdf_text(existing_pdfs)
    existing_chunks = get_text_chunks(existing_text)
    existing_vectorstore = get_vectorstore(existing_chunks) if existing_chunks else None

    st.header("Proposal Exploration Tool :books:")

    # Upload and process the user's new proposal requirements
    uploaded_pdf = st.file_uploader("Upload your PDF to define new proposal requirements", type=['pdf'])
    if uploaded_pdf:
        user_uploaded_text = get_pdf_text([uploaded_pdf])
        user_uploaded_chunks = get_text_chunks(user_uploaded_text)
        uploaded_vectorstore = get_vectorstore(user_uploaded_chunks) if user_uploaded_chunks else None

        if uploaded_vectorstore:
            # Initialize the conversation chain with existing proposals' content
            conversation_chain = initialize_conversation(existing_vectorstore) if existing_vectorstore else None

            st.subheader("Ask a Question")
            user_question = st.text_input("Ask a question about your document based on the existing proposals:")
            if user_question and conversation_chain:
                handle_userinput(conversation_chain, user_question)
        else:
            st.error("No valid text extracted from the uploaded PDF. Please check your document.")

if __name__ == '__main__':
    main()
