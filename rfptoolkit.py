import os
import streamlit as st
import requests
from io import BytesIO
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from datetime import datetime
import base64
import re

GITHUB_REPO_URL_UNDERGRAD = "https://api.github.com/repos/scooter7/ask-multiple-pdfs/contents/Undergrad"
GITHUB_REPO_URL_GRAD = "https://api.github.com/repos/scooter7/ask-multiple-pdfs/contents/Grad"
GITHUB_HISTORY_URL = "https://api.github.com/repos/scooter7/ask-multiple-pdfs/contents/ProposalChatHistory"

css = """
<style>
    .chat-container {
        max-width: 600px;
        margin: 0 auto;
        padding: 20px;
        background: #f9f9f9;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .chat-message {
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .user-message {
        background: #e0f7fa;
    }
    .bot-message {
        background: #ffe0b2;
    }
</style>
"""

KEYWORDS = [
    "website redesign", "SEO", "search engine optimization", "CRM", "Slate",
    "enrollment marketing", "recruitment marketing", "digital ads", "online advertising",
    "PPC", "social media", "surveys", "focus groups", "market research", "creative development",
    "graphic design", "video production", "brand redesign", "logo", "microsite",
    "landing page", "digital marketing", "predictive modeling", "financial aid optimization",
    "email marketing", "text message", "sms", "student search", "branding"
]

def main():
    st.set_page_config(
        page_title="Proposal Toolkit",
        page_icon="https://raw.githubusercontent.com/scooter7/ask-multiple-pdfs/main/ACE_92x93.png"
    )
    
    hide_toolbar_css = """
    <style>
        .css-14xtw13.e8zbici0 { display: none !important; }
    </style>
    """
    st.markdown(hide_toolbar_css, unsafe_allow_html=True)

    st.write(css, unsafe_allow_html=True)
    header_html = """
    <div style="text-align: center;">
        <h1 style="font-weight: bold;">Proposal Toolkit</h1>
        <img src="https://www.carnegiehighered.com/wp-content/uploads/2021/11/Twitter-Image-2-2021.png" alt="Icon" style="height:200px; width:500px;">
        <p align="left">Find and develop proposal resources. The text entry field will appear momentarily.</p>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

    if 'conversation_chain' not in st.session_state:
        st.session_state.conversation_chain = None
    if 'metadata' not in st.session_state:
        st.session_state.metadata = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'uploaded_pdf_text' not in st.session_state:
        st.session_state.uploaded_pdf_text = None
    if 'institution_name' not in st.session_state:
        st.session_state.institution_name = None
    if 'pdf_keywords' not in st.session_state:
        st.session_state.pdf_keywords = []

    uploaded_pdf = st.file_uploader("Upload an RFP PDF", type="pdf")
    if uploaded_pdf is not None:
        rfp_text = extract_text_from_pdf(uploaded_pdf)
        if rfp_text:
            st.session_state.uploaded_pdf_text = rfp_text
            st.session_state.institution_name = extract_institution_name(rfp_text)
            summarized_scope, extracted_keywords = summarize_scope_of_work(rfp_text)
            st.session_state.pdf_keywords = extracted_keywords
            st.subheader("Summarized Scope of Work")
            st.write(summarized_scope)
    
    undergrad_selected = st.checkbox("Undergraduate")
    grad_selected = st.checkbox("Graduate")
    
    pdf_docs, text_docs = get_github_docs(undergrad_selected, grad_selected)
    if pdf_docs or text_docs:
        raw_text, sources = get_docs_text(pdf_docs, text_docs)
        if st.session_state.uploaded_pdf_text:
            raw_text = st.session_state.uploaded_pdf_text + raw_text
            sources = [('Uploaded PDF', '')] + sources
        text_chunks = get_text_chunks(raw_text, sources)
        if text_chunks:
            vectorstore, metadata = get_vectorstore(text_chunks)
            st.session_state.conversation_chain = get_conversation_chain(vectorstore)
            st.session_state.metadata = metadata

    user_question = st.text_input("Find past RFP content and craft new content.")
    if user_question:
        handle_userinput(user_question, st.session_state.pdf_keywords, st.session_state.metadata)

def get_github_docs(undergrad_selected, grad_selected):
    github_token = st.secrets["github"]["access_token"]
    headers = {
        'Accept': 'application/vnd.github.v3+json',
        'Authorization': f'token {github_token}'
    }
    
    pdf_docs = []
    text_docs = []
    
    if undergrad_selected:
        pdf_docs.extend(fetch_docs_from_github(GITHUB_REPO_URL_UNDERGRAD, headers))
    if grad_selected:
        pdf_docs.extend(fetch_docs_from_github(GITHUB_REPO_URL_GRAD, headers))
    
    return pdf_docs, text_docs

def fetch_docs_from_github(repo_url, headers):
    response = requests.get(repo_url, headers=headers)
    if response.status_code != 200:
        st.error(f"Failed to fetch files: {response.status_code}, {response.text}")
        return []
    
    files = response.json()
    if not isinstance(files, list):
        st.error(f"Unexpected response format: {files}")
        return []
    
    pdf_docs = []
    text_docs = []
    
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
    
    return pdf_docs, text_docs

def get_docs_text(pdf_docs, text_docs):
    text = ""
    sources = []
    for pdf, source in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text() or ""
                text += page_text
                sources.append(source)
        except PdfReadError:
            st.error(f"Failed to read PDF file: {source}")
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
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    retriever = vectorstore.as_retriever()
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)
    return conversation_chain

def extract_text_from_pdf(uploaded_pdf):
    try:
        pdf_reader = PdfReader(uploaded_pdf)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            text += page_text
        return text
    except PdfReadError as e:
        st.error(f"Failed to read the PDF file: {e}")
        return None

def extract_institution_name(text):
    institution_name = ""
    for line in text.split('\n'):
        if "college" in line.lower() or "university" in line.lower():
            institution_name = line.strip()
            break
    return institution_name

def summarize_scope_of_work(text):
    keyword_summary = {keyword: [] for keyword in KEYWORDS}
    proposal_deadline = ""
    submission_method = ""

    for line in text.split('\n'):
        for keyword in KEYWORDS:
            if re.search(rf'\b{keyword}\b', line, re.IGNORECASE):
                keyword_summary[keyword].append(line)
        if re.search(r'\b(deadline|due date)\b', line, re.IGNORECASE):
            proposal_deadline = line
        if re.search(r'\b(submission|submit|sent via)\b', line, re.IGNORECASE):
            submission_method = line

    summary = ["**Scope of Work:**"]
    for keyword, occurrences in keyword_summary.items():
        if occurrences:
            summary.append(f"- **{keyword.capitalize()}:** {', '.join(occurrences)}")
    if proposal_deadline:
        summary.append(f"- **Proposal Deadline:** {proposal_deadline}")
    if submission_method:
        summary.append(f"- **Submission Method:** {submission_method}")

    extracted_keywords = [keyword for keyword, occurrences in keyword_summary.items() if occurrences]
    return '\n'.join(summary), extracted_keywords

def modify_response_language(original_response, institution_name):
    response = original_response.replace(" they ", " we ")
    response = response.replace("They ", "We ")
    response = response.replace(" their ", " our ")
    response = response.replace("Their ", "Our ")
    response = response.replace(" them ", " us ")
    response = response.replace("Them ", "Us ")
    if institution_name:
        response = response.replace("the current opportunity", institution_name)
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

def handle_userinput(user_question, pdf_keywords, metadata):
    if 'conversation_chain' in st.session_state and st.session_state.conversation_chain:
        conversation_chain = st.session_state.conversation_chain

        # Combine user question with the keywords extracted from the PDF
        query = f"{user_question}. Keywords: {', '.join(pdf_keywords)}. Leverage content from the selected Grad or Undergrad folder to match these keywords and pull contextually relevant information, including in-depth scope of work and pricing when available."
        
        response = conversation_chain({'question': query})
        st.session_state.chat_history = response['chat_history']
        institution_name = st.session_state.institution_name
        for i, message in enumerate(st.session_state.chat_history):
            modified_content = modify_response_language(message.content, institution_name)
            if i % 2 == 0:
                st.write(f'<div class="chat-message user-message">{modified_content}</div>', unsafe_allow_html=True)
            else:
                # Get citations for this response
                citations = []
                for doc in response.get('source_documents', []):
                    index = response['source_documents'].index(doc)
                    source = metadata[index]
                    citations.append(f"Source: {source}")
                citations_text = "\n".join(citations)
                st.write(f'<div class="chat-message bot-message">{modified_content}\n\n{citations_text}</div>', unsafe_allow_html=True)
        save_chat_history(st.session_state.chat_history)
    else:
        st.error("The conversation model is not initialized. Please wait until the model is ready.")

if __name__ == '__main__':
    main()
