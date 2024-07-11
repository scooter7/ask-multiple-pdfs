import os
import streamlit as st
import requests
from io import BytesIO
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from datetime import datetime
import base64

GITHUB_REPO_URL = "https://api.github.com/repos/scooter7/ask-multiple-pdfs/contents/enrollmentbestpractices"
GITHUB_HISTORY_URL = "https://api.github.com/repos/scooter7/ask-multiple-pdfs/contents/enrollmentbestpracticeshistory"

def main():
    # Set page config
    st.set_page_config(
        page_title="Enrollment Nest Practice Bot",
        page_icon="https://raw.githubusercontent.com/scooter7/ask-multiple-pdfs/main/ACE_92x93.png"
    )
    
    # Hide the Streamlit toolbar
    hide_toolbar_css = """
    <style>
        .css-14xtw13.e8zbici0 { display: none !important; }
    </style>
    """
    st.markdown(hide_toolbar_css, unsafe_allow_html=True)

    st.write(css, unsafe_allow_html=True)
    header_html = """
    <div style="text-align: center;">
        <h1 style="font-weight: bold;">Enrollment Best Practices Bot</h1>
        <img src="https://www.carnegiehighered.com/wp-content/uploads/2021/11/Twitter-Image-2-2021.png" alt="Icon" style="height:200px; width:500px;">
        <p align="left">Hey there! Ask about enrollment best practices. The text entry field will appear momentarily.</p>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        
    pdf_docs = get_github_pdfs()
    if pdf_docs:
        raw_text = get_pdf_text(pdf_docs)
        text_chunks, chunk_metadata = get_text_chunks(raw_text)
        if text_chunks:
            vectorstore = get_vectorstore(text_chunks, chunk_metadata)
            st.session_state.conversation = get_conversation_chain(vectorstore)
    
    user_question = st.text_input("Ask about enrollment best practices")
    if user_question:
        handle_userinput(user_question)

def get_github_pdfs():
    github_token = st.secrets["github"]["access_token"]
    headers = {
        'Accept': 'application/vnd.github.v3+json',
        'Authorization': f'token {github_token}'
    }
    response = requests.get(GITHUB_REPO_URL, headers=headers)
    if response.status_code != 200:
        st.error(f"Failed to fetch files: {response.status_code}, {response.text}")
        return []
    files = response.json()
    if not isinstance(files, list):
        st.error(f"Unexpected response format: {files}")
        return []
    pdf_docs = []
    for file in files:
        if 'name' in file and file['name'].endswith('.pdf'):
            pdf_url = file.get('download_url')
            if pdf_url:
                response = requests.get(pdf_url, headers=headers)
                pdf_docs.append({'file': BytesIO(response.content), 'source': file['name']})
    return pdf_docs

def get_pdf_text(pdf_docs):
    text = ""
    source_metadata = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf['file'])
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text() or ""
            text += page_text
            source_metadata.append({'text': page_text, 'source': f"{pdf['source']} - Page {page_num + 1}"})
    return text, source_metadata

def get_text_chunks(text, metadata):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text)
    chunk_metadata = [metadata[i // 1000] for i in range(len(chunks))]  # Match chunks with their metadata
    return chunks, chunk_metadata

def get_vectorstore(text_chunks, chunk_metadata):
    if not text_chunks:
        raise ValueError("No text chunks available for embedding.")
    os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, metadata=chunk_metadata, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)
    return conversation_chain

def modify_response_language(original_response, citations):
    response = original_response.replace(" they ", " we ")
    response = response.replace("They ", "We ")
    response = response.replace(" their ", " our ")
    response = response.replace("Their ", "Our ")
    response = response.replace(" them ", " us ")
    response = response.replace("Them ", "Us ")
    if citations:
        response += "\n\nSources:\n" + "\n".join(f"- {citation}" for citation in citations)
    return response

def save_chat_history(chat_history):
    github_token = st.secrets["github"]["access_token"]
    headers = {
        'Accept': 'application/vnd.github.v3+json',
        'Authorization': f'token {github_token}'
    }
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"chat_history_{date_str}.txt"
    chat_content = "\n\n".join(f"{'User:' if i % 2 == 0 else 'Bot:'} {message.content}" for i, message in enumerate(chat_history))
    
    encoded_content = base64.b64encode(chat_content.encode('utf-8')).decode('utf-8')
    data = {
        "message": f"Save chat history on {date_str}",
        "content": encoded_content,
        "branch": "main"
    }
    response = requests.put(f"{GITHUB_HISTORY_URL}/{file_name}", headers=headers, json=data)
    if response.status_code == 201:
        st.success("Chat history saved successfully.")
    else:
        st.error(f"Failed to save chat history: {response.status_code}, {response.text}")

def handle_userinput(user_question):
    if 'conversation' in st.session_state and st.session_state.conversation:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']
        citations = [msg.metadata['source'] for msg in response['source_documents']]
        for i, message in enumerate(st.session_state.chat_history):
            modified_content = modify_response_language(message.content, citations if i % 2 != 0 else None)
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", modified_content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", modified_content), unsafe_allow_html=True)
        # Save chat history after each interaction
        save_chat_history(st.session_state.chat_history)
    else:
        st.error("The conversation model is not initialized. Please wait until the model is ready.")

if __name__ == '__main__':
    main()
