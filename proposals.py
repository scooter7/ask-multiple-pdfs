import os
import streamlit as st
import requests
from PyPDF2 import PdfReader
from io import BytesIO
import re
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

GITHUB_REPO_URL = "https://github.com/scooter7/ask-multiple-pdfs/tree/main/rfps/"

def get_github_pdfs(repo_url):
    api_url = "https://api.github.com/repos/scooter7/ask-multiple-pdfs/contents/rfps"
    headers = {'Accept': 'application/vnd.github.v3+json'}
    response = requests.get(api_url, headers=headers)
    files = response.json()
    
    if isinstance(files, dict):  # Handle potential error messages
        st.write("Error fetching files:", files.get("message", "Unknown error"))
        return []

    pdf_docs = []
    for file in files:
        try:
            if file.get('name', '').endswith('.pdf'):
                pdf_url = file['download_url']
                response = requests.get(pdf_url)
                pdf_docs.append(BytesIO(response.content))
        except Exception as e:
            print(f"Failed processing file {file}: {e}")
    
    return pdf_docs

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            text += page_text
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
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        modified_content = modify_response_language(message.content)
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", modified_content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", modified_content), unsafe_allow_html=True)

def extract_key_sections(text, section_keywords):
    sections = {}
    for keyword in section_keywords:
        # Improved pattern to capture up to the next major section or end of content
        pattern = re.compile(rf'(?s)\b{re.escape(keyword)}\b(.*?)(?=\n\w|\Z)', re.IGNORECASE)
        matches = pattern.findall(text)
        if matches:
            sections[keyword] = ' '.join(matches[0].splitlines())
    return sections

def summarize_section(section_text):
    # Summarize by extracting key sentences or the first paragraph
    sentences = re.split(r'(?<=[.!?])\s+', section_text.strip(), maxsplit=4)
    summary = ' '.join(sentences[:4])  # Limit summary to first 4 sentences
    if len(summary) > 500:
        return summary[:500] + "..."  # Limit length to 500 characters for brevity
    return summary

def estimate_budget(section_text):
    # Improved pattern to find all monetary values including those with commas and decimals
    budget_pattern = re.compile(r'\$\s*[\d,]+\.?\d*')
    budgets = budget_pattern.findall(section_text.replace(',', ''))
    budget_values = [float(b.replace('$', '').replace(' ', '').replace(',', '')) for b in budgets]
    total_budget = sum(budget_values)
    return total_budget

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

    uploaded_file = st.file_uploader("Upload a PDF file to add to the knowledge base", type=["pdf"])
    if uploaded_file:
        user_pdf_docs = [BytesIO(uploaded_file.read())]
        user_text = get_pdf_text(user_pdf_docs)
        key_sections = extract_key_sections(user_text, ["Scope of Work", "Budget", "Objectives", "Requirements"])

        if key_sections:
            st.write("### Key Sections and Summaries")
            for key, section in key_sections.items():
                st.write(f"**{key}:**")
                st.write(summarize_section(section))
            if "Budget" in key_sections:
                budget = estimate_budget(key_sections["Budget"])
                st.write(f"**Estimated Total Budget: ${budget:,.2f}**")
            else:
                st.write("**Estimated Total Budget: $0.00**")
        else:
            st.write("No key sections found in the uploaded document.")
            st.write("**Estimated Total Budget: $0.00**")

        combined_text = user_text + "\n" + get_pdf_text(get_github_pdfs(GITHUB_REPO_URL))
    else:
        github_pdf_docs = get_github_pdfs(GITHUB_REPO_URL)
        combined_text = get_pdf_text(github_pdf_docs)

    if combined_text:
        text_chunks = get_text_chunks(combined_text)
        if text_chunks:
            vectorstore = get_vectorstore(text_chunks)
            st.session_state.conversation = get_conversation_chain(vectorstore)
            st.write("Conversation model is ready to answer questions.")
        else:
            st.write("No text chunks found. Please check the content of your PDFs.")
    else:
        st.write("No PDF documents found. Please upload a file or check the GitHub repo.")

    user_question = st.text_input("Ask CAI about anything Carnegie:")
    if user_question:
        handle_userinput(user_question)

if __name__ == '__main__':
    main()
