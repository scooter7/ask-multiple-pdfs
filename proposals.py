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

# Existing function definitions or imports here...

def get_github_pdfs(repo_url):
    # Correctly format the API URL to list files under the 'docs' directory
    api_url = "https://api.github.com/repos/scooter7/ask-multiple-pdfs/contents/docs"
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

def main():
    st.set_page_config(page_title="Proposal Exploration Tool", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    st.header("Proposal Exploration Tool :books:")

    # Use the get_github_pdfs function to load PDFs from the 'rfps' folder
    knowledge_pdfs = get_github_pdfs("https://github.com/scooter7/ask-multiple-pdfs/contents/rfps")
    knowledge_text = get_pdf_text(knowledge_pdfs)
    knowledge_chunks = get_text_chunks(knowledge_text)

    # Display the knowledge base size
    st.write(f"Loaded knowledge from **{len(knowledge_pdfs)}** RFP documents in the 'rfps' folder.")

    # Upload and process the user's new proposal requirements
    uploaded_pdf = st.file_uploader("Upload your PDF to define new proposal requirements", type=['pdf'])
    if uploaded_pdf:
        user_uploaded_text = get_pdf_text([uploaded_pdf])
        user_uploaded_chunks = get_text_chunks(user_uploaded_text)

        # Create a combined context from both the uploaded document and the existing knowledge
        combined_text = knowledge_text + "\n" + user_uploaded_text
        combined_chunks = get_text_chunks(combined_text)
        combined_vectorstore = get_vectorstore(combined_chunks) if combined_chunks else None

        if combined_vectorstore:
            # Initialize the conversation chain with the combined context
            combined_conversation_chain = get_conversation_chain(combined_vectorstore)

            st.subheader("Ask About the Uploaded Document")
            user_question = st.text_input("What do you want to know about the uploaded document?")

            # Directly use uploaded document's context for specific questions
            if user_question and combined_conversation_chain:
                st.subheader("Responses Based on the Uploaded Document")
                handle_userinput(combined_conversation_chain, user_question)

            st.subheader("Ask How Existing Knowledge Applies")
            knowledge_question = st.text_input("How can the existing knowledge be applied here?")

            # Use the combined context to answer how existing knowledge can be applied
            if knowledge_question and combined_conversation_chain:
                st.subheader("Application of Existing Knowledge")
                handle_userinput(combined_conversation_chain, knowledge_question)
        else:
            st.error("No valid text extracted from the uploaded PDF. Please check your document.")

if __name__ == '__main__':
    main()
