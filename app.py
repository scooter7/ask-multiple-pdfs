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
import io
from langchain.callbacks import get_openai_callback
from tiktoken import encoding_for_model

# Function to count tokens using OpenAI tiktoken tokenizer
def count_tokens(text, model_name="gpt-3.5-turbo"):
    enc = encoding_for_model(model_name)
    return len(enc.encode(text))

# Token limit for the model (gpt-3.5-turbo and gpt-4 have different limits)
MODEL_TOKEN_LIMIT = 16385
CHUNK_SIZE = 300  # Reduce chunk size to avoid large text overflow

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=CHUNK_SIZE, chunk_overlap=100, length_function=len)
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

def limit_conversation_history(conversation_history, max_tokens=5000):
    """Limit the conversation history to avoid exceeding the token limit."""
    if conversation_history is None:
        return []
    
    current_token_count = sum([count_tokens(message.content) for message in conversation_history])
    limited_history = conversation_history

    while current_token_count > max_tokens and limited_history:
        # Remove the oldest message
        limited_history = limited_history[1:]
        current_token_count = sum([count_tokens(message.content) for message in limited_history])

    return limited_history

def handle_userinput(user_question):
    # Check token length of user input
    if count_tokens(user_question) > 3000:  # Adjust this based on model limit
        st.error("Your question is too long. Please shorten it.")
        return
    
    # Limit conversation history tokens
    st.session_state.chat_history = limit_conversation_history(st.session_state.chat_history)

    # Process the user question
    response = st.session_state.conversation({'question': user_question})

    # Ensure response has a valid chat history
    if 'chat_history' in response:
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template2.replace("{{MSG}}", message.content), unsafe_allow_html=True)

            # Download bot responses
            if i % 2 != 0:  # Only for bot responses
                text_to_download = message.content
                st.download_button(
                    label="Download Response",
                    data=io.StringIO(text_to_download).getvalue(),
                    file_name="bot_response.txt",
                    mime="text/plain"
                )
    else:
        st.error("No valid response from the model.")

def main():
    st.set_page_config(page_title="Document Exploration Tool", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # Initialize session state variables if they don't exist
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Document Exploration Tool :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                # Explicitly check total tokens of all chunks
                total_tokens = sum(count_tokens(chunk) for chunk in text_chunks)
                if total_tokens > MODEL_TOKEN_LIMIT:
                    st.error("The document is too large. Please upload a smaller document or split the file.")
                else:
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()
