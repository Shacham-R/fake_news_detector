import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Fake News Detector", page_icon=":rolled_up_newspaper:", layout="centered")

tab1, tab2, tab3 = st.tabs(["Overview", "Process", "Demo"])


with tab1:
    st.header("Overview")
    st.image("streamlit_app_data/troll1")
    if st.button("click me!"):
        st.snow()
    else:
        None
    image
        
    
with tab2:
    st.header("Process")

with tab3:
    st.header("Demo")
    st.warning("This is a prototype, plaese do not take seriously")
    url = st.text_input('Please enter a url to test')
    text = st.text_area("Article Text","Lolram Oopsam")