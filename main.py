import streamlit as st
st.set_page_config(page_title="main")
st.title("뉴스기사 주제 추천 솔루션")
st.write("뉴스기사 주제를 추천해주는 솔루션입니다.")

category = ["정치", "경제", "사회", "생활/문화", "세계", "기술/IT", "연예", "스포츠"]

### LOAD MODEL ###
from newsrec.model import load_model
from newsrec.tokenizer import CustomTokenizer

model_path = "models/news_supervised_kr"
model = load_model(model_path)

### SET CUSTOM LABEL ###
mappings = model.topic_mapper_.get_mappings()
mappings = {value : category[key] for key, value in mappings.items()}
model.set_topic_labels(mappings)

st.session_state["model"] = model
st.session_state["category"] = category

