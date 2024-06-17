from src.ner_inference import NER_Pipeline
from src.tsa_inference import TSA_Pipeline

import streamlit as st

import re

dict_tone = {1: 'негативно', 2: 'умеренно негативно', 3: 'нейтрально', 4: 'умеренно позитивно', 5: 'позитивно'}

def clear_text(text: str) -> str:
    text = re.sub(r'\?{2,}', '', text)
    text = re.sub(r'\u200b', '', text)
    text = re.sub(r'\\[^ ]*', '', text)
    text = re.sub(r'\@\S{2,}', '', text)
    text = re.sub(r'http\S{2,}', '', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'\"{2,}', '"', text)
    return text


NER_MODEL_NAME = 'model_xlm_9864.pickle'
TSA_MODEL_NAME = 'tsa_rubertmodel_balanced1.pickle'

ner = NER_Pipeline(model_name=NER_MODEL_NAME)
tsa = TSA_Pipeline(model_name=TSA_MODEL_NAME)

st.sidebar.subheader('О приложении')
st.sidebar.write('Анализ тональности текста относительно компаний, представленых в нём')

# start the user interface
st.title("Определение тональности")
st.write("Введите текст поста и нажмите на кнопку: GO")

text = st.text_area("Введите текст", "Здесь ваш текст", max_chars=5000, key='to_GO')

if st.button('GO', key='GO'):
    text = clear_text(text)
    companies = ner.get_companies(text)

    st.write(f"Извлечённые компании: {', '.join(companies)}\n\n")

    for company in companies:
        sentiment = tsa.get_sentiment(text=text, company_name=company)
        st.write(f"Тональность относительно компании {company} равна {sentiment} - {dict_tone[sentiment]}")
