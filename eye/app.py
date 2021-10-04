from UI import home, test, SessionState
import streamlit as st


session_state = SessionState.get(his=None, model_obj=None)


PAGES = {"home": home, "test": test}
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]


if selection == "home":
    session_state.his, session_state.model_obj = home.app()
elif selection == "test":
    if session_state.model_obj is not None:
        test.app(session_state.model_obj)
    else:
        st.write("first biuld and train model in home page")
