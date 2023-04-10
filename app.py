import numpy as np
import streamlit as st
from transformers import pipeline

import torch

def bertweet(data):
    specific_model = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")
    result = specific_model(data)
    label = result[0]['label']
    score = result[0]['score']

    return label, score 

def roberta(data):
    specific_model = pipeline(model="cardiffnlp/twitter-roberta-base-sentiment")
    result = specific_model(data)
    label = result[0]['label']
    score = result[0]['score']

    if(label == 'LABEL_0'):
        label = 'Negative'
    elif(label == 'LABEL_1'):
        label = 'Neutral'
    else:
        label = 'Positive'

    return label, score 

def getSent(data, model):
    if(model == 'Bertweet'):
        label,score = bertweet(data)
        col1, col2 = st.columns(2)
        col1.metric("Feeling",label,None)
        col2.metric("Score",score,None)
    elif(model == 'Roberta'):
        label,score = roberta(data)
        col1, col2 = st.columns(2)
        col1.metric("Feeling",label,None)
        col2.metric("Score",score,None)

def rendPage():
    st.title("Sentiment Analysis")
    userText = st.text_input('User Input', "Hope you are having a great day!")
    st.text("")
    type = st.selectbox(
        'Choose your model',
        ('Bertweet','Roberta',))
    st.text("")

    if st.button('Calculate'):
        if(userText!="" and type != None):
            st.text("")
            getSent(userText,type)

rendPage()