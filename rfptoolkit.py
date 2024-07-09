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
from langchain.chains.question_answering import load_qa_chain
from htmlTemplates import css, bot_template, user_template
from datetime import datetime
import base64

GITHUB_REPO_URL_UNDERGRAD = "https://api.github.com/repos/scooter7/ask-multiple-pdfs/contents/Undergrad"
GITHUB_REPO_URL_GRAD = "https://api.github.com/repos/scooter7/ask-multiple-pdfs/contents/Grad"
GITHUB_HISTORY_URL = "https://api.github.com/repos/scooter7/ask-multiple-pdfs/contents/ProposalChatHistory"

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

    if 'conversation_chain' not in st.session_state:
        st.session_state.conversation_chain = None
    if 'metadata' not in st.session_state:
        st.session_state.metadata = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    undergrad_selected = st.checkbox("Undergraduate")
    grad_selected = st.checkbox("Graduate")
    
    pdf_docs, text_docs = get_github_docs(undergrad_selected, grad_selected)
    if pdf_docs or text_docs:
        raw_text, sources = get_docs_text(pdf_docs, text_docs)
        text_chunks = get_text_chunks(raw_text, sources)
        if text_chunks:
            vectorstore, metadata = get_vectorstore(text_chunks)
            st.session_state.conversation_chain = get_conversation_chain(vectorstore)
            st.session_state.metadata = metadata
    
    # Upload PDF and Summarize Scope of Work
    uploaded_pdf = st.file_uploader("Upload an RFP PDF", type="pdf")
    if uploaded_pdf is not None:
        rfp_text = extract_text_from_pdf(uploaded_pdf)
        if rfp_text:
            summarized_scope = summarize_scope_of_work(rfp_text)
            st.subheader("Summarized Scope of Work")
            st.write(summarized_scope)
    
    user_question = st.text_input("Ask about enrollment best practices")
    if user_question:
        handle_userinput(user_question)

def get_github_docs(undergrad_selected, grad_selected):
    github_token = st.secrets["github"]["access_token"]
    headers = {
        'Accept': 'application/vnd.github.v3+json',
        'Authorization': f'token {github_token}'
    }
    
    pdf_docs = []
    text_docs = []
    
    if undergrad_selected:
        pdf_docs.extend(fetch_docs_from_github(GITHUB_REPO_URL_UNDERGRAD, headers, pdf_docs, text_docs))
    if grad_selected:
        pdf_docs.extend(fetch_docs_from_github(GITHUB_REPO_URL_GRAD, headers, pdf_docs, text_docs))
    
    return pdf_docs, text_docs

def fetch_docs_from_github(repo_url, headers, pdf_docs, text_docs):
    response = requests.get(repo_url, headers=headers)
    if response.status_code != 200:
        st.error(f"Failed to fetch files: {response.status_code}, {response.text}")
        return []
    
    files = response.json()
    if not isinstance(files, list):
        st.error(f"Unexpected response format: {files}")
        return []
    
    for file in files:
        if 'name' in file:
            if file['name'].endswith('.pdf'):
                pdf_url = file.get('download_url')
                if pdf_url:
                    response = requests.get(pdf_url, headers=headers)
                    pdf_docs.append((BytesIO(response.content), file['name']))
            elif file['name'].endswith('.txt'):
                text_url = file.get('download_url')
                if text_url:
                    response = requests.get(text_url, headers=headers)
                    text_docs.append((response.text, file['name']))
    
    return pdf_docs

def get_docs_text(pdf_docs, text_docs):
    text = ""
    sources = []
    for pdf, source in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            text += page_text
            sources.append(source)
    for doc, source in text_docs:
        text += doc
        sources.append(source)
    return text, sources

def get_text_chunks(text, sources):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text)
    return [(chunk, sources[i % len(sources)]) for i, chunk in enumerate(chunks)]

def get_vectorstore(text_chunks):
    if not text_chunks:
        raise ValueError("No text chunks available for embedding.")
    os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
    embeddings = OpenAIEmbeddings()
    texts, metadata = zip(*text_chunks)
    vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings)
    return vectorstore, metadata

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    retriever = vectorstore.as_retriever()
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)
    return conversation_chain

def extract_text_from_pdf(uploaded_pdf):
    try:
        pdf_reader = PdfReader(uploaded_pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"Failed to read the PDF file: {e}")
        return None

def summarize_scope_of_work(text):
    try:
        os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
        llm = ChatOpenAI()
        qa_chain = load_qa_chain(llm, chain_type="map_reduce")
        summary = qa_chain({"question": "Summarize the scope of work.", "input_documents": [text]})
        return summary['answer']
    except Exception as e:
        st.error(f"Failed to summarize the scope of work: {e}")
        return None

def modify_response_language(original_response):
    response = original_response.replace(" they ", " we ")
    response = original_response.replace("They ", "We ")
    response = original_response.replace(" their ", " our ")
    response = original_response.replace("Their ", "Our ")
    response = original_response.replace(" them ", " us ")
    response = original_response.replace("Them ", "Us ")
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
    if 'conversation_chain' in st.session_state and st.session_state.conversation_chain:
        conversation_chain = st.session_state.conversation_chain
        response = conversation_chain({'question': user_question})
        st.session_state.chat_history = response['chat_history']
        metadata = st.session_state.metadata
        for i, message in enumerate(st.session_state.chat_history):
            modified_content = modify_response_language(message.content)
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", modified_content), unsafe_allow_html=True)
            else:
                # Get citations for this response
                citations = []
                for doc in response.get('source_documents', []):
                    index = response['source_documents'].index(doc)
                    citations.append(f"Source: {metadata[index]}")
                citations_text = "\n".join(citations)
                st.write(bot_template.replace("{{MSG}}", f"{modified_content}\n\n{citations_text}"), unsafe_allow_html=True)
        # Save chat history after each interaction
        save_chat_history(st.session_state.chat_history)
    else:
        st.error("The conversation model is not initialized. Please wait until the model is ready.")

if __name__ == '__main__':
    main()
