try:
    import streamlit.ReportThread as ReportThread
    from streamlit.server.Server import Server
except Exception:
    # Streamlit >= 0.65.0
    import streamlit.report_thread as ReportThread
    from streamlit.server.server import Server

class SessionState:
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

def get(**kwargs):
    ctx = ReportThread.get_report_ctx()
    current_server = Server.get_current()
    session_infos = current_server._session_info_by_id.values()
    
    for session_info in session_infos:
        s = session_info.session
        if s == ctx.session:
            this_session = s
    
    if not hasattr(this_session, '_custom_session_state'):
        this_session._custom_session_state = SessionState(**kwargs)
    
    return this_session._custom_session_state
