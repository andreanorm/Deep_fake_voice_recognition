import streamlit as st
from PIL import Image
import base64

st.set_page_config(page_title="Our Team", page_icon="🧑🏽‍🎓")

st.markdown("<h1 style='text-align: center; '>Who are we?</h1>", unsafe_allow_html=True)

st.markdown(
    """
    ### Our team is from Le Wagon - Batch #1429.
    """)

columns_0 = st.columns(2)

columns_0[0].markdown(
    """
    We bring together diverse expertise, eager to apply **the power of data science in the realm of audio**.""")
columns_0[0].markdown(
    """
    **Andrea** brings a wealth of experience from the world of business, marketing and operations, offering a strategic edge to our endeavors.""")
columns_0[0].markdown(
    """
    **Elise**, with her rich background as an Industrial Engineer in Aerospace, infuses innovation and technical precision.""")
columns_0[0].markdown(
    """
    **Nicole**, armed with a Master’s in Metallurgical Engineering, adds a layer of analytical depth and scientific rigor.""")
columns_0[0].markdown(
    """
    **Youssef**, formerly an accountant specializing in taxes, rounds out our team with a keen eye for detail, logic and accuracy. Together, we are a complementary team, driven to explore the uncharted territories of audio data science.
    """
)

columns_0[1].image("images/IMG_5326.jpg")
