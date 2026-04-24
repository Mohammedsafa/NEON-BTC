import streamlit as st

def initialize_session_state():
    defaults = {
        'step': 1,
        'processed_data': None,
        'is_processed': False,
        'current_diff': 0,
        'apply_log': False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def reset_state():
    st.session_state.is_processed = False
    st.session_state.processed_data = None
    st.session_state.step = 1
