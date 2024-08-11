import os
import asyncio
import streamlit as st
from httpx_oauth.clients.google import GoogleOAuth2
from session_state import get
from io import BytesIO
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from datetime import datetime
import base64
import httpx

GITHUB_REPO_URL = "https://api.github.com/repos/scooter7/ask-multiple-pdfs/contents/docs"
GITHUB_HISTORY_URL = "https://api.github.com/repos/scooter7/ask-multiple-pdfs/contents/History"

async def get_authorization_url(client, redirect_uri):
    authorization_url = await client.get_authorization_url(
        redirect_uri,
        scope=["profile", "email"],
        extras_params={"access_type": "offline"},
    )
    return authorization_url

async def get_access_token(client, redirect_uri, code):
    token = await client.get_access_token(code, redirect_uri)
    return token

async def get_user_info(client, token):
    try:
        user_info_endpoint = "https://www.googleapis.com/oauth2/v1/userinfo"
        headers = {
            "Authorization": f"Bearer {token['access_token']}"
        }
        async with httpx.AsyncClient() as async_client:
            response = await async_client.get(user_info_endpoint, headers=headers)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        st.error(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
        raise e
    except Exception as e:
        st.error(f"An error occurred: {e}")
        raise e

def get_github_pdfs():
    github_token = st.secrets["github"]["access_token"]
    headers = {
        'Accept': 'application/vnd.github.v3+json',
        'Authorization': f'token {github_token}'
    }
    try:
        response = httpx.get(GITHUB_REPO_URL, headers=headers)
        response.raise_for_status()
        files = response.json()
        if not isinstance(files, list):
            st.error(f"Unexpected response format: {files}")
            return []
        pdf_docs = []
        for file in files:
            if 'name' in file and file['name'].endswith('.pdf'):
                pdf_url = file.get('download_url')
                if pdf_url:
                    response = httpx.get(pdf_url, headers=headers)
                    response.raise_for_status()
                    pdf_docs.append(BytesIO(response.content))
        return pdf_docs
    except httpx.HTTPStatusError as e:
        st.error(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
        return []
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return []

def get_pdf_text(pdf_docs):
    text = []
    metadata = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
                metadata.append({'source': f"{pdf} - Page {page_num + 1}"})  # Example metadata
    return text, metadata

def get_text_chunks(text, metadata):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = []
    chunk_metadata = []
    for i, page_text in enumerate(text):
        page_chunks = text_splitter.split_text(page_text)
        chunks.extend(page_chunks)
        chunk_metadata.extend([metadata[i]] * len(page_chunks))  # Assign correct metadata to each chunk
    return chunks, chunk_metadata

def get_vectorstore(text_chunks, chunk_metadata):
    if not text_chunks:
        raise ValueError("No text chunks available for embedding.")
    os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
    embeddings = OpenAIEmbeddings()
    documents = [Document(page_content=chunk, metadata=chunk_metadata[i]) for i, chunk in enumerate(text_chunks)]
    vectorstore = FAISS.from_documents(documents, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=vectorstore.as_retriever(), 
        memory=memory, 
        return_source_documents=True
    )
    return conversation_chain

def modify_response_language(original_response, citations=None):
    response = original_response.replace(" they ", " we ")
    response = response.replace("They ", "We ")
    response = response.replace(" their ", " our ")
    response = response.replace("Their ", "Our ")
    response = response.replace(" them ", " us ")
    response = response.replace("Them ", "Us ")
    if citations:
        response += "\n\nSources:\n" + "\n".join(
            f"- [{citation}](https://github.com/scooter7/ask-multiple-pdfs/blob/main/docs/{citation.split(' - ')[0]})"
            for citation in citations)
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
    try:
        response = httpx.put(f"{GITHUB_HISTORY_URL}/{file_name}", headers=headers, json=data)
        response.raise_for_status()
        st.success("Chat history saved successfully.")
    except httpx.HTTPStatusError as e:
        st.error(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

def handle_userinput(user_question):
    if 'conversation' in st.session_state and st.session_state.conversation:
        # Call the conversation chain
        response = st.session_state.conversation({'question': user_question})
        
        # Store the chat history
        st.session_state.chat_history = response['chat_history']

        # Extract the answer and source documents from the response
        answer = response['answer']
        source_documents = response.get('source_documents', [])

        # Debug: Print out the source documents to check metadata
        st.write("Source Documents:", source_documents)

        # Extract citations from source documents
        citations = []
        for doc in source_documents:
            metadata = doc.metadata
            if metadata and 'source' in metadata:
                citation = metadata['source']
                citations.append(citation)
        
        # Debug: Print out the citations
        st.write("Citations:", citations)

        # Modify the response with hyperlinks
        modified_content = modify_response_language(answer, citations)
        
        # Display the conversation
        st.write(bot_template.replace("{{MSG}}", modified_content), unsafe_allow_html=True)
        
        save_chat_history(st.session_state.chat_history)
    else:
        st.error("The conversation model is not initialized. Please wait until the model is ready.")

def main():
    st.set_page_config(
        page_title="Ask Carnegie Everything",
        page_icon="https://raw.githubusercontent.com/scooter7/ask-multiple-pdfs/main/ACE_92x93.png"
    )
    
    hide_toolbar_css = """
    <style>
        .css-14xtw13.e8zbici0 { display: none !important; }
    </style>
    """
    st.markdown(hide_toolbar_css, unsafe_allow_html=True)
    
    client_id = st.secrets["google_auth"]["client_id"]
    client_secret = st.secrets["google_auth"]["client_secret"]
    redirect_uri = st.secrets["google_auth"]["redirect_uris"][0]

    client = GoogleOAuth2(client_id, client_secret)
    authorization_url = asyncio.run(get_authorization_url(client, redirect_uri))

    session_state = get(token=None, user_id=None, user_email=None)

    if session_state.token is None:
        try:
            code = st.experimental_get_query_params()['code'][0]
        except KeyError:
            st.markdown(f'[Authorize with Google]({authorization_url})')
        else:
            try:
                token = asyncio.run(get_access_token(client, redirect_uri, code))
                session_state.token = token
                user_info = asyncio.run(get_user_info(client, token))
                session_state.user_id = user_info['id']
                session_state.user_email = user_info['email']
                st.experimental_rerun()
            except Exception as e:
                st.write(f"Error fetching token: {e}")
                st.markdown(f'[Authorize with Google]({authorization_url})')
    else:
        st.write(f"You're logged in as {session_state.user_email}")
        if st.button("Log out"):
            session_state.token = None
            session_state.user_id = None
            session_state.user_email = None
            st.experimental_rerun()

        st.write(css, unsafe_allow_html=True)
        header_html = """
        <div style="text-align: center;">
            <h1 style="font-weight: bold;">Ask Carnegie Everything - ACE</h1>
            <img src="https://www.carnegiehighered.com/wp-content/uploads/2021/11/Twitter-Image-2-2021.png" alt="Icon" style="height:200px; width:500px;">
            <p align="left">Hey there! Just a quick heads-up: while I'm here to jazz up your day and be super helpful, keep in mind that I might not always have the absolute latest info or every single detail nailed down. So, if you're making big moves or crucial decisions, it's always a good idea to double-check with your manager or division lead, HR, or those cool cats on the operations team. And hey, if you run into any hiccups or just wanna shoot the breeze, hit me up anytime! Your feedback is like fuel for this chatbot engine, so don't hold back—give <a href="https://form.asana.com/?k=6rnnec7Gsxzz55BMqpp6ug&d=654504412089816">the suggestions and feedback form </a>a whirl! The text entry field will appear momentarily.</p>
        </div>
        """
        st.markdown(header_html, unsafe_allow_html=True)
        if 'conversation' not in st.session_state:
            st.session_state.conversation = None
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        pdf_docs = get_github_pdfs()
        if pdf_docs:
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            if text_chunks:
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)
        user_question = st.text_input("Ask ACE about anything Carnegie:")
        if user_question:
            handle_userinput(user_question)

if __name__ == '__main__':
    main()
