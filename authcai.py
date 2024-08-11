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

def modify_response_language(original_response, citations=None):
    response = original_response.replace(" they ", " we ")
    response = response.replace("They ", "We ")
    response = response.replace(" their ", " our ")
    response = response.replace("Their ", "Our ")
    response = response.replace(" them ", " us ")
    response = response.replace("Them ", "Us ")
    
    if citations and len(citations) > 0:
        response += "\n\nSources:\n" + "\n".join(
            f"- [{citation}](https://github.com/scooter7/gemini_multipdf_chat/blob/main/docs/{citation.split(' - ')[0]})"
            for citation in citations
        )
    
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

def handle_userinput():
    # Display the conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Process new user input
    if prompt := st.chat_input():
        # Append user input to session messages
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Split the user input into chunks if necessary
        query_chunks = chunk_query(prompt)
        full_response = ''
        all_citations = []

        # Process each chunk of the query
        for chunk in query_chunks:
            response = user_input(chunk)
            full_response += ''.join(response['output_text'])
            all_citations.extend(response.get('citations', []))

        # Modify the response to include citations and apply any language changes
        modified_response = modify_response_language(full_response, all_citations)

        # Display and store the assistant's response
        with st.chat_message("assistant"):
            st.write(modified_response)
            st.session_state.messages.append({"role": "assistant", "content": modified_response})

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

        if 'messages' not in st.session_state:
            st.session_state.messages = []

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

        handle_userinput()

if __name__ == '__main__':
    main()
