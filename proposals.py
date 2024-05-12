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
import json

# Utilize existing functions...
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
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

    st.header("Proposal Exploration Tool :books:")

    # Fetch and process the existing PDFs from the GitHub 'rfps' directory
    knowledge_pdfs = get_github_pdfs("https://github.com/scooter7/ask-multiple-pdfs")
    knowledge_text = get_pdf_text(knowledge_pdfs)
    knowledge_chunks = get_text_chunks(knowledge_text)

    st.write(f"Loaded knowledge from **{len(knowledge_pdfs)}** RFP documents in the 'rfps' folder.")

    # Process the user's new proposal requirements
    uploaded_pdf = st.file_uploader("Upload your PDF to define new proposal requirements", type=['pdf'])
    if uploaded_pdf:
        user_uploaded_text = get_pdf_text([uploaded_pdf])
        user_uploaded_chunks = get_text_chunks(user_uploaded_text)

        # Combine contexts from both uploaded and existing knowledge
        combined_text = knowledge_text + "\n" + user_uploaded_text
        combined_chunks = get_text_chunks(combined_text)
        combined_vectorstore = get_vectorstore(combined_chunks) if combined_chunks else None

        if combined_vectorstore:
            combined_conversation_chain = get_conversation_chain(combined_vectorstore)

            st.subheader("Ask About the Uploaded Document")
            user_question = st.text_input("What do you want to know about the uploaded document?")

            if user_question and combined_conversation_chain:
                st.subheader("Responses Based on the Uploaded Document")
                handle_userinput(combined_conversation_chain, user_question)

            st.subheader("Ask How Existing Knowledge Applies")
            knowledge_question = st.text_input("How can the existing knowledge be applied here?")

            if knowledge_question and combined_conversation_chain:
                st.subheader("Application of Existing Knowledge")
                handle_userinput(combined_conversation_chain, knowledge_question)
        else:
            st.error("No valid text extracted from the uploaded PDF. Please check your document.")

if __name__ == '__main__':
    main()
