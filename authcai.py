import os
import requests
from io import BytesIO
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
import streamlit as st
from langchain_community.vectorstores import FAISS
from openai import OpenAI
from dotenv import load_dotenv
from langchain.schema import Document
from datetime import datetime
import base64

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Function to get list of PDFs from GitHub repository
def get_pdfs_from_github():
    api_url = "https://api.github.com/repos/scooter7/ask-multiple-pdfs/contents/docs"
    headers = {"Accept": "application/vnd.github.v3+json"}
    response = requests.get(api_url, headers=headers)
    if response.status_code == 200:
        files = response.json()
        pdf_files = [file['download_url'] for file in files if file['name'].endswith('.pdf')]
        return pdf_files
    else:
        st.error(f"Failed to fetch list of PDF files from GitHub: {response.status_code}")
        return []

# Function to download PDF files from the GitHub repository
def download_pdfs_from_github():
    pdf_urls = get_pdfs_from_github()
    pdf_docs = []
    for url in pdf_urls:
        response = requests.get(url)
        if response.status_code == 200:
            file_name = url.split('/')[-1]
            with open(file_name, 'wb') as f:
                f.write(response.content)
            pdf_docs.append(file_name)
        else:
            st.error(f"Failed to download {url}")
    return pdf_docs

# Read all PDF files and return text
def get_pdf_text(pdf_docs):
    text = []
    source_metadata = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
                source_metadata.append({'source': f"{pdf} - Page {page_num + 1}", 'url': f"https://github.com/scooter7/ask-multiple-pdfs/blob/main/docs/{pdf}"})
            else:
                st.warning(f"Failed to extract text from page in {pdf}")
    return text, source_metadata

# Split text into chunks
def get_text_chunks(text, metadata, chunk_size=2000, chunk_overlap=500):
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    chunk_metadata = []
    for i, page_text in enumerate(text):
        page_chunks = splitter.split_text(page_text)
        chunks.extend(page_chunks)
        chunk_metadata.extend([metadata[i]] * len(page_chunks))  # Assign correct metadata to each chunk
    return chunks, chunk_metadata

# Get embeddings for each chunk
def get_vector_store(chunks, metadata):
    if not chunks:
        st.error("No text chunks available for embedding")
        return None
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    documents = [Document(page_content=chunk, metadata=metadata[i]) for i, chunk in enumerate(chunks)]
    vector_store = FAISS.from_documents(documents, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def load_or_create_vector_store(chunks, metadata):
    if os.path.exists("faiss_index"):
        try:
            embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
            vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            return vector_store
        except Exception as e:
            st.error(f"Failed to load FAISS index: {e}")
    return get_vector_store(chunks, metadata)

# Function to handle user input, process it, and return citations with the response
def user_input(user_question, max_retries=5, delay=2):
    vector_store = load_or_create_vector_store([], [])
    if not vector_store:
        st.error("Failed to load or create the vector store.")
        return {"output_text": ["Failed to load or create the vector store."]}

    try:
        docs = vector_store.similarity_search(user_question)
    except Exception as e:
        st.error(f"Failed to perform similarity search: {e}")
        return {"output_text": [f"Failed to perform similarity search: {e}"]}

    response_text = ""
    citations = []

    for doc in docs:
        context = doc.page_content
        context_chunks = get_text_chunks([context], [doc.metadata])

        for chunk, meta in zip(context_chunks[0], context_chunks[1]):
            for attempt in range(max_retries):
                try:
                    completion = client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You specialize in leveraging the PDF content in the repository to find and discuss information from these documents."},
                            {"role": "user", "content": f"Context: {chunk}\n\nQuestion: {user_question}"}
                        ]
                    )
                    response_text += completion.choices[0].message.content + " "
                    citations.append(meta['source'])
                    break
                except Exception as e:
                    if 'Resource has been exhausted' in str(e):
                        st.warning(f"API quota has been exhausted. Retrying in {delay} seconds...")
                        time.sleep(delay)
                        delay *= 2  # Exponential backoff
                    else:
                        st.error(f"Failed to generate response: {e}")
                        return {"output_text": [f"Failed to generate response: {e}"]}

    return {"output_text": [response_text], "citations": citations}

# Modify response to include citations
def modify_response_language(original_response, citations):
    response = original_response.replace(" they ", " we ")
    response = response.replace("They ", "We ")
    response = response.replace(" their ", " our ")
    response = response.replace("Their ", "Our ")
    response = response.replace(" them ", " us ")
    response = response.replace("Them ", "Us ")
    if citations:
        response += "\n\nSources:\n" + "\n".join(f"- [{citation}](https://github.com/scooter7/ask-multiple-pdfs/blob/main/docs/{citation.split(' - ')[0]})" for citation in citations)
    return response

def main():
    st.set_page_config(
        page_title="Leverage Existing Proposal Content",
    )

    # Automatically download and process PDFs from GitHub
    with st.spinner("Downloading and processing PDFs..."):
        pdf_docs = download_pdfs_from_github()
        if pdf_docs:
            raw_text, source_metadata = get_pdf_text(pdf_docs)
            if raw_text:
                text_chunks, chunk_metadata = get_text_chunks(raw_text, source_metadata)
                if text_chunks:
                    vector_store = load_or_create_vector_store(text_chunks, chunk_metadata)
                    if vector_store:
                        st.success("PDF processing complete")
                    else:
                        st.error("Failed to create vector store")
                else:
                    st.error("No text chunks created")
            else:
                st.error("No text extracted from PDFs")
        else:
            st.error("No PDFs downloaded")

    # Main content area for displaying chat messages
    st.title("Find and engage with past proposal content")
    st.write("Welcome to the chat!")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # Chat input
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Find and engage with past Carnegie proposals."}]

    for message in st.session_state.messages:
        st.write(message["content"])

    if prompt := st.text_input("Ask ACE about anything Carnegie:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.write(prompt)

        # Split the prompt into smaller chunks and process each one
        query_chunks = chunk_query(prompt)
        full_response = ''
        all_citations = []

        for chunk in query_chunks:
            response = user_input(chunk)
            for item in response['output_text']:
                full_response += item
            all_citations.extend(response['citations'])

        modified_response = modify_response_language(full_response, all_citations)

        st.write(modified_response)
        st.session_state.messages.append({"role": "assistant", "content": modified_response})

if __name__ == "__main__":
    main()
