import time

import pandas as pd
import plotly.express as px
import streamlit as st
import requests
import os
KEY = os.environ.get("KEY")
API_URL = "https://api-inference.huggingface.co/models/DaniilOr/ce_sentiment"
headers = {"Authorization": f"Bearer {KEY}"}
DISABLED = False

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


def check(text):
    output = query({
        "inputs": text,
    })
    print(output)
    if 'error' in output:
        st.write("Model is loading, please, wait")
        DISABLED = True
        time.sleep(
            30
        )
        check(text)
    DISABLED = False
    output = output[0]
    sentiments = {}
    for i in output:
        if i['label'] == 'LABEL_0':
            sentiments['neutral'] = i['score']
        elif i['label'] == 'LABEL_1':
            sentiments['negative'] = i['score']
        elif i['label'] == "LABEL_2":
            sentiments['positive'] = i['score']
    print(pd.DataFrame.from_dict(sentiments, orient='index'))
    fig = px.bar(pd.DataFrame.from_dict(sentiments, orient='index'),
                 x=pd.DataFrame.from_dict(sentiments, orient='index').index,
                 y=pd.DataFrame.from_dict(sentiments, orient='index')[0],
                 color=pd.DataFrame.from_dict(sentiments, orient='index').index,
                 category_orders={'x': ['negative', 'neutral', 'positive']},
                 title='Scores distribution',)
    fig.update_layout(
        xaxis=dict(
            ticktext=['negative', 'neutral', 'positive'],
        ),
        yaxis=dict(
            range=[0, 1]
        ),
        xaxis_title=dict(text='Sentiment', font=dict(size=16, )),
        yaxis_title=dict(text='Probability', font=dict(size=16, )),
    )

    st.plotly_chart(fig)


if __name__ == "__main__":
    text = st.text_area("This tool allows sentiment analysis (on the scale positive, neutral, negative) of various opinions about 
    Circular Economy in the Construction Industry. If you would like to check one, please enter the text in the box below. 
    Please make sure it does not exceed 200 words. Should you have any questions, please address them to aidana.tleuken@nu.edu.kz. 
    This webpage is prepared in the framework of research article [add webpage after published]",)
    btn = st.button("Click me", on_click=check, args=(text,), disabled=DISABLED)
