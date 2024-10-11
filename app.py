import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template2, user_template
import io  # Add for downloading text

# Helper function to count tokens in a string
def count_tokens(text):
    return len(text.split())

# Function to truncate chat history
def truncate_chat_history(chat_history, max_tokens=3000):
    total_tokens = 0
    truncated_history = []
    
    # Traverse from the latest to earliest messages
    for message in reversed(chat_history):
        tokens_in_message = count_tokens(message.content)
        if total_tokens + tokens_in_message <= max_tokens:
            truncated_history.insert(0, message)  # Add to the beginning of the list
            total_tokens += tokens_in_message
        else:
            break
    
    return truncated_history

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

# Adjust chunk size to stay within token limits
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=100, length_function=len)
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

def handle_userinput(user_question):
    conversation = st.session_state.conversation
    chat_history = st.session_state.chat_history or []

    # Ensure conversation object is available
    if conversation is None:
        st.error("Please upload and process documents first.")
        return

    # Truncate chat history to avoid exceeding token limits
    truncated_history = truncate_chat_history(chat_history)
    st.session_state.chat_history = truncated_history
    
    response = conversation({'question': user_question})
    truncated_history.extend(response['chat_history'])
    st.session_state.chat_history = truncated_history

    for i, message in enumerate(truncated_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template2.replace("{{MSG}}", message.content), unsafe_allow_html=True)

        # Add button to download the last bot response as a text file
        if i % 2 != 0:  # Only for bot responses
            text_to_download = message.content
            st.download_button(
                label="Download Response",
                data=io.StringIO(text_to_download).getvalue(),
                file_name="bot_response.txt",
                mime="text/plain"
            )

# Handle large inputs by checking token count before processing
def handle_large_input(user_question):
    max_token_limit = 16384
    question_tokens = count_tokens(user_question)
    
    if question_tokens > max_token_limit:
        st.error("The input is too long. Please shorten the question.")
    else:
        handle_userinput(user_question)

def main():
    st.set_page_config(page_title="Document Exploration Tool", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Document Exploration Tool :books:")
    user_question = st.text_input("Ask a question about your documents:")

    if user_question:
        handle_large_input(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                if text_chunks:
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                else:
                    st.error("No valid text extracted from PDFs. Please check your documents.")

if __name__ == '__main__':
    main()
