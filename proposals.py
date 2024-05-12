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

def convert_github_url_to_api_url(web_url):
    path_part = web_url.split("github.com/")[1]
    segments = path_part.split("/")
    username = segments[0]
    repo = segments[1]
    if "tree" in segments and "main" in segments:
        path_index = segments.index("main") + 1
        folder_path = "/".join(segments[path_index:])
    else:
        folder_path = ""
    api_url = f"https://api.github.com/repos/{username}/{repo}/contents/{folder_path}"
    return api_url

def get_github_pdfs(web_repo_url):
    api_url = convert_github_url_to_api_url(web_repo_url)
    headers = {'Accept': 'application/vnd.github.v3+json'}
    response = requests.get(api_url, headers=headers)
    if response.status_code != 200:
        return []
    files = response.json()
    pdf_docs = []
    for file in files:
        if file.get('name', '').endswith('.pdf'):
            pdf_url = file.get('download_url', '')
            if pdf_url:
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

def handle_userinput(user_question):
    if 'conversation' in st.session_state and st.session_state.conversation:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']
        for i, message in enumerate(st.session_state.chat_history):
            modified_content = modify_response_language(message.content)
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", modified_content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", modified_content), unsafe_allow_html=True)
    else:
        st.write("The conversation model is not initialized due to missing RFP data.")

def analyze_requirements(requirements_pdf, rfps_vectorstore):
    requirements_text = get_pdf_text([requirements_pdf])
    requirements_chunks = get_text_chunks(requirements_text)
    requirements_store = get_vectorstore(requirements_chunks)
    best_matches = rfps_vectorstore.search(requirements_store, k=3)
    results = []
    for match in best_matches:
        results.append(" ".join(match['text']))
    return results

def main():
    st.set_page_config(page_title="CAI", page_icon="https://www.carnegiehighered.com/wp-content/uploads/2021/11/Twitter-Image-2-2021.png")
    st.write(css, unsafe_allow_html=True)
    header_html = """
    <div style="text-align: center;">
        <h1 style="font-weight: bold;">Carnegie Artificial Intelligence - CAI</h1>
        <img src="https://www.carnegiehighered.com/wp-content/uploads/2021/11/Twitter-Image-2-2021.png" alt="Icon" style="height:200px; width:500px;">
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)
    rfps_vectorstore = None
    GITHUB_REPO_URL = "https://github.com/scooter7/ask-multiple-pdfs/tree/main/rfps/"
    rfps_docs = get_github_pdfs(GITHUB_REPO_URL)
    if rfps_docs:
        rfps_text = get_pdf_text(rfps_docs)
        rfps_chunks = get_text_chunks(rfps_text)
        rfps_vectorstore = get_vectorstore(rfps_chunks)
        st.session_state.conversation = get_conversation_chain(rfps_vectorstore)
    else:
        st.write("No RFP documents were retrieved from the repository.")
        st.session_state.conversation = None
    uploaded_file = st.file_uploader("Upload a requirements PDF", type="pdf")
    if uploaded_file and rfps_vectorstore:
        uploaded_pdf = BytesIO(uploaded_file.getvalue())
        results = analyze_requirements(uploaded_pdf, rfps_vectorstore)
        st.write("### Matching Content from RFPs:")
        for result in results:
            st.write(result)
    elif uploaded_file:
        st.write("Unable to analyze requirements; no RFP data is available.")
    user_question = st.text_input("Ask CAI about anything Carnegie:")
    if user_question:
        handle_userinput(user_question)

if __name__ == '__main__':
    main()
