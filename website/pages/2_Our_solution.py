
import joblib
import streamlit as st
import requests
import matplotlib as plt
import shap
import pandas as pd
from data import *
import json

st.set_option('deprecation.showPyplotGlobalUse', False)


st.set_page_config(
            page_title="Audio Anthenticator", # => Quick reference - Streamlit
            page_icon="🔊",
            layout="centered", # wide
            initial_sidebar_state="auto") # collapsed

st.markdown("<h1 style='text-align: center; '>AudioAuthenticator</h1>", unsafe_allow_html=True)

st.markdown("""
Artificial intelligence tools have given scammers a potent weapon for trying to trick people into sending them money. Our innovative solution is designed to counter this threat. It empowers you to identify and thwart these fraudulent attempts

✅ Stay one step ahead of scammers. Test our tool now and experience the power of AI in protecting your security:

1. **Upload Your Audio**: Select the audio file you're unsure about.

2. Instant Analysis: Click 'Predict' to let our advanced AI **evaluate the audio**.

3. **Understand the Verdict**: After the prediction, our tool not only tells you if the audio is 'fake' or 'real'
""")

tabs_ = st.tabs(["Predict", "Convert"])

with tabs_[0]:
    uploaded_file = st.file_uploader("Choose an audio file to analyze with the AI model:", key=1)
    url = 'https://audioauthenticator-fy2s5a5k2q-ew.a.run.app/predict/'

    # st.markdown("<h3 style='text-align: center; color: white;'>Deep fake detection </h3>", unsafe_allow_html=True)

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/ogg')

        if st.button('Predict the audio'):
            # print is visible in the server output, not in the page
            print('button clicked!')
            with st.spinner(text='In progress'):
                response = requests.post(url, files={'file': uploaded_file.getvalue()}).json(cls=json.JSONDecoder)
                st.success('Done')
            pred = response["prediction"]
            proba = round(response["probability"]*100,2)

            columns_0 = st.columns(2)
            columns_0[0].write('The audio seems to be:')
            columns_0[0].write(pred)
            columns_0[1].write('With a probability of:')
            if proba > 0.5 :
                ind_ = 'Likely'
            if proba > 0.70:
                ind_ = 'Highly Likely'
            if proba > 0.9:
                ind_ = 'Almost Certain'

            columns_0[1].write(f"{ind_} ")


            "Most impactfull features : "
            with st.status("Loading figure:"):
                    st.write("Getting features...")
                    X_demo =  pd.DataFrame(get_features(uploaded_file,'test'),
                                           columns=['chroma_stft', 'rms', 'spectral_centroid',
                                                    'spectral_bandwidth','rolloff',
                                                    'zero_crossing_rate', 'mfcc1', 'mfcc2',
                                                    'mfcc3', 'mfcc4','mfcc5', 'mfcc6',
                                                    'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11',
                                                    'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16',
                                                    'mfcc17', 'mfcc18','mfcc19', 'mfcc20'])
                    st.write("Loading model..")
                    model = joblib.load('model/last_XGB')
                    st.write("initializing figure..")
                    shap.initjs();
                    explainer = shap.Explainer(model)
                    shap_values_one = explainer(X_demo)
                    shap.plots.waterfall(shap_values_one[0])
                    st.pyplot(bbox_inches='tight')

with tabs_[1]:
    "Real voice"
    uploaded_file2 = st.file_uploader("Choose an audio file", key=2)

    # st.markdown("<h3 style='text-align: center; color: white;'>Deep fake detection </h3>", unsafe_allow_html=True)

    if uploaded_file2 is not None:
        st.audio(uploaded_file2, format='audio/ogg')

    uploaded_file3 = st.file_uploader("Choose an audio file", key=3)

    # st.markdown("<h3 style='text-align: center; color: white;'>Deep fake detection </h3>", unsafe_allow_html=True)
    'Converted voice'
    if uploaded_file3 is not None:
        st.audio(uploaded_file3, format='audio/ogg')
