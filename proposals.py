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
import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree

def get_continuous_chunks(text):
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    continuous_chunk = []
    current_chunk = []
    for i in chunked:
        if type(i) == Tree:
            current_chunk.append(" ".join([token for token, pos in i.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continue
    if current_chunk:
        named_entity = " ".join(current_chunk)
        if named_entity not in continuous_chunk:
            continuous_chunk.append(named_entity)

    return continuous_chunk

nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def fetch_pdfs_from_github(github_url):
    response = requests.get(github_url)
    try:
        data = response.json()
        if isinstance(data, list):
            pdf_urls = [file['download_url'] for file in data if file['name'].endswith('.pdf')]
            return pdf_urls
        else:
            st.error("Unexpected response structure from GitHub API.")
            print("Response from GitHub API:", json.dumps(data, indent=2))
            return []
    except json.JSONDecodeError:
        st.error("Failed to decode the response from GitHub.")
        print("Response content:", response.text)
        return []

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

def get_text_chunks(text, chunk_size=1500, chunk_overlap=300):
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

    st.header("Proposal Exploration Tool :books:")

    github_url = "https://api.github.com/repos/scooter7/ask-multiple-pdfs/contents/rfps"
    pdf_urls = fetch_pdfs_from_github(github_url)
    knowledge_pdfs = download_pdfs(pdf_urls)
    knowledge_text = get_pdf_text(knowledge_pdfs)
    knowledge_chunks = get_text_chunks(knowledge_text)

    uploaded_pdf = st.file_uploader("Upload your PDF to define new proposal requirements", type=['pdf'])
    if uploaded_pdf:
        user_uploaded_text = get_pdf_text([uploaded_pdf])
        user_uploaded_chunks = get_text_chunks(user_uploaded_text)

        user_entities = get_continuous_chunks(user_uploaded_text)
        user_doc_summary = ' '.join(user_entities)

        combined_text = f"Past proposals:\n{knowledge_text}\n\nUser-uploaded document summary:\n{user_doc_summary}\n\nFull text from the uploaded document:\n{user_uploaded_text}"
        combined_chunks = get_text_chunks(combined_text)
        combined_vectorstore = get_vectorstore(combined_chunks) if combined_chunks else None

        if combined_vectorstore:
            conversation_chain = initialize_conversation(combined_vectorstore)

            st.subheader("Ask a Question About How to Address New Requirements Using Past Proposals")
            user_question = st.text_input("Enter your question about how to address new requirements using past proposals:")

            if user_question:
                st.subheader("Responses Based on Combined Knowledge")
                handle_userinput(conversation_chain, user_question)
        else:
            st.error("No valid text extracted from the uploaded PDF. Please check your document.")
    else:
        st.warning("Please upload a document to start the analysis.")

if __name__ == '__main__':
    main()
