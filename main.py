import streamlit as st
st.set_page_config(page_title="main")
st.title("주제 추천 솔루션")
st.write("텍스트의 주제를 추천해주는 솔루션입니다.")

### LOAD MODEL ###
from pathlib import Path
from topicrec.model import load_model
from topicrec.tokenizer import CustomTokenizer
from topicrec.utils import set_custom_label

root_path = str(Path(__file__).parent) + "/"

option = st.selectbox(
    "어떤 텍스트의 주제를 추천받고 싶으신가요?",
    ("뉴스", "특허")
)
st.write("선택한 텍스트는 : ", option)

if option == "뉴스":
    data_type = "news"
    model_path = root_path + "models/news_supervised_kr"
    data_path = root_path + "data/newsData/"
    st.session_state["data_path"] = data_path
else:
    data_type = "patent"
    model_path = root_path + "models/patent_supervised_kr"
    train_data_path = root_path + "data/patData/1.Training"
    test_data_path = root_path + "data/patData/2.Validation"
    st.session_state["train_data_path"] = train_data_path
    st.session_state["test_data_path"] = test_data_path

### SET CUSTOM LABEL ###
model = load_model(model_path)
model, org_mappings, mappings, category = set_custom_label(model, data_type=data_type)

inv_org_mappings = {}
for k, v in org_mappings.items():
    inv_org_mappings[v] = k

st.session_state["model"] = model
st.session_state["category"] = category
st.session_state["data_type"] = data_type
st.session_state["mappings"] = mappings
st.session_state["org_mappings"] = org_mappings
st.session_state["inv_org_mappings"] = inv_org_mappings

