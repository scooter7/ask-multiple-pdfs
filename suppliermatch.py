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
from langchain.schema import Document
from datetime import datetime
import base64
import re
import numpy as np

GITHUB_REPO_URL_SUPPLIERS = "https://api.github.com/repos/scooter7/ask-multiple-pdfs/contents/suppliers"
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
        font-family: Arial, sans-serif;
        word-wrap: break-word;
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
    "email marketing", "text message", "sms", "student search", "branding", "pricing"
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
        <img src="https://github.com/scooter7/ask-multiple-pdfs/blob/main/silogo.jpg?raw=true" alt="Icon" style="height:200px; width:500px;">
        <p align="left">Match your RFP to Suppliers.</p>
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
    
    supplier_query = st.text_input("Enter service areas to find matching suppliers")
    if supplier_query:
        handle_supplier_query(supplier_query)

def fetch_docs_from_github(repo_url, headers):
    response = requests.get(repo_url, headers=headers)
    if response.status_code != 200:
        st.error(f"Failed to fetch files: {response.status_code}, {response.text}")
        return []
    
    files = response.json()
    if not isinstance(files, list):
        st.error(f"Unexpected response format: {files}")
        return []
    
    docs = []
    for file in files:
        if 'name' in file:
            if file['name'].endswith('.pdf'):
                pdf_url = file.get('download_url')
                if pdf_url:
                    response = requests.get(pdf_url, headers=headers)
                    docs.append((BytesIO(response.content), file['name'], file['html_url']))
            elif file['name'].endswith('.csv'):
                csv_url = file.get('download_url')
                if csv_url:
                    response = requests.get(csv_url, headers=headers)
                    docs.append((response.content.decode('utf-8'), file['name'], file['html_url']))
    
    return docs

def get_docs_text(docs):
    text = ""
    sources = []
    for doc, source, url in docs:
        if isinstance(doc, BytesIO):
            pdf_reader = PdfReader(doc)
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text() or ""
                text += page_text
                sources.append({'source': source, 'page': page_num + 1, 'url': url})
        else:
            text += doc
            sources.append({'source': source, 'page': None, 'url': url})
    return text, sources

def get_text_chunks(text, sources):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text)
    chunk_metadata = []
    for i, chunk in enumerate(chunks):
        source_info = sources[i % len(sources)]
        if source_info['page'] is not None:
            chunk_metadata.append(f"{source_info['source']} - Page {source_info['page']} [{source_info['url']}]")
        else:
            chunk_metadata.append(f"{source_info['source']} [{source_info['url']}]")
    return chunks, chunk_metadata

def get_vectorstore(text_chunks, chunk_metadata):
    if not text_chunks:
        raise ValueError("No text chunks available for embedding.")
    os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
    embeddings = OpenAIEmbeddings()
    documents = [Document(page_content=chunk, metadata={'source': chunk_metadata[i]}) for i, chunk in enumerate(text_chunks)]
    vectorstore = FAISS.from_documents(documents, embedding=embeddings)
    return vectorstore, chunk_metadata

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory, return_source_documents=True)
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

def handle_userinput(user_input, pdf_keywords):
    if 'conversation_chain' in st.session_state and st.session_state.conversation_chain:
        conversation_chain = st.session_state.conversation_chain

        # Stage 1: Retrieve broader set of relevant documents
        initial_retrieval = conversation_chain.retriever.get_relevant_documents(user_input)
        
        # Stage 2: Rerank the retrieved documents
        reranked_documents = rerank_documents(initial_retrieval, user_input)
        
        # Create a custom input for the conversation chain
        response = run_conversation_chain(conversation_chain, user_input, reranked_documents)
        st.session_state.chat_history = response['chat_history']
        metadata = st.session_state.metadata
        institution_name = st.session_state.institution_name

        # Consolidate the bot response
        final_response = ""
        citations = []
        for message in response['chat_history']:
            modified_content = modify_response_language(message.content, institution_name)
            final_response += modified_content + "\n\n"

        for doc in response['source_documents']:
            citations.append(f"{doc.metadata['source']} - Page {doc.metadata.get('page', 'N/A')}")

        citations_text = "\n".join(set(citations))  # Remove duplicates
        st.write(f'<div class="chat-message bot-message">{final_response}\n\n{citations_text}</div>', unsafe_allow_html=True)
        save_chat_history(st.session_state.chat_history)
    else:
        st.error("The conversation model is not initialized. Please wait until the model is ready.")

def handle_supplier_query(query):
    suppliers = get_supplier_docs()
    if suppliers:
        raw_text, sources = get_docs_text(suppliers)
        text_chunks, chunk_metadata = get_text_chunks(raw_text, sources)
        if text_chunks:
            vectorstore, metadata = get_vectorstore(text_chunks, chunk_metadata)
            st.session_state.metadata = metadata
            supplier_conversation_chain = get_conversation_chain(vectorstore)
            response = run_conversation_chain(supplier_conversation_chain, query, text_chunks)
            final_response = ""
            citations = []
            for message in response['chat_history']:
                final_response += message.content + "\n\n"

            for doc in response['source_documents']:
                citations.append(f"{doc.metadata['source']} - Page {doc.metadata.get('page', 'N/A')}")

            citations_text = "\n".join(set(citations))  # Remove duplicates
            st.write(f'<div class="chat-message bot-message">{final_response}\n\n{citations_text}</div>', unsafe_allow_html=True)

def get_supplier_docs():
    github_token = st.secrets["github"]["access_token"]
    headers = {
        'Accept': 'application/vnd.github.v3+json',
        'Authorization': f'token {github_token}'
    }
    return fetch_docs_from_github(GITHUB_REPO_URL_SUPPLIERS, headers)

def run_conversation_chain(chain, question, documents):
    # Prepare context from documents
    context = ' '.join([doc.page_content for doc in documents if isinstance(doc, Document)])
    # Combine question and context into a single input
    combined_input = f"Question: {question}\n\nContext: {context}"
    return chain({'question': combined_input})

def rerank_documents(documents, query):
    # Get embeddings for the query
    embeddings = OpenAIEmbeddings()
    query_embedding = np.array(embeddings.embed_query(query)).reshape(1, -1)
    
    # Get embeddings for the documents
    document_embeddings = np.array([np.array(embeddings.embed_query(doc.page_content)) for doc in documents if isinstance(doc, Document)])
    
    # Calculate cosine similarity between query and documents
    similarities = cosine_similarity(query_embedding, document_embeddings)[0]
    
    # Sort documents by similarity score
    sorted_indices = np.argsort(similarities)[::-1]
    reranked_documents = [documents[idx] for idx in sorted_indices]
    
    return reranked_documents

def cosine_similarity(query_embedding, document_embeddings):
    # Compute dot product between query and documents
    dot_product = np.dot(document_embeddings, query_embedding.T)
    
    # Compute norms (magnitudes) of query and documents
    query_norm = np.linalg.norm(query_embedding)
    doc_norms = np.linalg.norm(document_embeddings, axis=1)
    
    # Compute cosine similarity
    similarities = dot_product / (query_norm * doc_norms)
    
    return similarities

if __name__ == '__main__':
    main()
