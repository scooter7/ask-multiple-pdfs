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
import io  # For downloading text

# Patch langchain to avoid AttributeError with missing 'verbose'
import langchain
langchain.verbose = False  # Prevent AttributeError

# Function to extract text from uploaded PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

# Function to split the extracted text into smaller chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=100, length_function=len)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store from the text chunks
def get_vectorstore(text_chunks):
    if not text_chunks:
        raise ValueError("No text chunks available for embedding.")
    os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Function to create a conversation chain using LangChain
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(temperature=0.7)  # Adjust as needed
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Helper function to chunk user input if it's too long
def chunk_user_input(user_input, chunk_size=1000):
    words = user_input.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Function to handle user input and generate responses
def handle_userinput(user_question):
    # Ensure chat_history is initialized
    if st.session_state.chat_history is None:
        st.session_state.chat_history = []

    # Trim chat history to last 10 messages if necessary
    if len(st.session_state.chat_history) > 10:
        st.session_state.chat_history = st.session_state.chat_history[-10:]

    # Process the user question in chunks if necessary
    user_question_chunks = chunk_user_input(user_question)

    # Generate response using the conversation chain for each chunk
    for chunk in user_question_chunks:
        response = st.session_state.conversation({'question': chunk})
        st.session_state.chat_history = response['chat_history']

        # Display chat history
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template2.replace("{{MSG}}", message.content), unsafe_allow_html=True)

            # Download bot responses
            if i % 2 != 0:
                text_to_download = message.content
                st.download_button(
                    label="Download Response",
                    data=io.StringIO(text_to_download).getvalue(),
                    file_name="bot_response.txt",
                    mime="text/plain"
                )

# Main function that runs the Streamlit app
def main():
    st.set_page_config(page_title="Document Exploration Tool", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Document Exploration Tool :books:")
    user_question = st.text_input("Ask a question about your documents:")

    if st.button("Clear Session"):
        st.session_state.conversation = None
        st.session_state.chat_history = []
        st.success("Session cleared!")

    if user_question:
        if len(user_question.split()) > 3000:
            st.error("Your input is too long. Please shorten the question.")
        else:
            handle_userinput(user_question)

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
