import os
import streamlit as st
import requests
from PyPDF2 import PdfReader
from io import BytesIO
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

GITHUB_REPO_URL = "https://github.com/scooter7/ask-multiple-pdfs/tree/main/cx/"

def get_github_pdfs(repo_url):
    # Correctly format the API URL to list files under the 'cx' directory
    api_url = "https://api.github.com/repos/scooter7/ask-multiple-pdfs/contents/cx"
    headers = {'Accept': 'application/vnd.github.v3+json'}
    
    try:
        response = requests.get(api_url, headers=headers)
        # Check if the response was successful
        if response.status_code != 200:
            print(f"Failed to fetch files: {response.status_code}, {response.text}")
            return []
        
        files = response.json()
        
        if not isinstance(files, list):
            print(f"Unexpected response format: {files}")
            return []
        
        pdf_docs = []
        for file in files:
            if 'name' in file and file['name'].endswith('.pdf'):
                pdf_url = file.get('download_url')
                if pdf_url:
                    response = requests.get(pdf_url)
                    pdf_docs.append(BytesIO(response.content))
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
    
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
    # Simple replacements; could be expanded based on actual usage
    response = original_response.replace(" they ", " we ")
    response = response.replace("They ", "We ")
    response = response.replace(" their ", " our ")
    response = response.replace("Their ", "Our ")
    return response

def handle_userinput(user_question):
    if st.session_state.conversation is not None:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            modified_content = modify_response_language(message.content)
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", modified_content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", modified_content), unsafe_allow_html=True)
    else:
        st.error("The conversation model is not initialized. Please wait until the model is ready.")

def main():
    st.set_page_config(page_title="CXBot", page_icon="https://www.carnegiehighered.com/wp-content/uploads/2021/11/Twitter-Image-2-2021.png")
    st.write(css, unsafe_allow_html=True)

    header_html = """
    <div style="text-align: center;">
        <h1 style="font-weight: bold;">CX Bot</h1>
        <img src="https://www.carnegiehighered.com/wp-content/uploads/2021/11/Twitter-Image-2-2021.png" alt="Icon" style="height:200px; width:500px;">
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)
    
    # Ensure session state is initialized for conversation
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    
    # Retrieve PDFs from GitHub
    pdf_docs = get_github_pdfs(GITHUB_REPO_URL)
    if pdf_docs:
        raw_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(raw_text)
        if text_chunks:
            vectorstore = get_vectorstore(text_chunks)
            st.session_state.conversation = get_conversation_chain(vectorstore)

    user_question = st.text_input("Ask CXBot anything about finding and applying to the right colleges and universities:")
    if user_question:
        handle_userinput(user_question)

if __name__ == '__main__':
    main()
