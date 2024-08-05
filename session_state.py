import streamlit as st

class SessionState:
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

def get(**kwargs):
    session = st.session_state
    if not hasattr(session, '_custom_session_state'):
        session._custom_session_state = SessionState(**kwargs)
    return session._custom_session_state
