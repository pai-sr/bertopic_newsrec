import streamlit as st
st.set_page_config(page_title="main")
st.title("주제 추천 솔루션")
st.write("텍스트의 주제를 추천해주는 솔루션입니다.")

### LOAD MODEL ###
from newsrec.model import load_model
from newsrec.tokenizer import CustomTokenizer
from newsrec.utils import set_custom_label

root_path = "/mnt/prj/BERTopic/"


model_path = root_path + "models/patent_supervised_kr"
train_data_path = root_path + "data/patData/1.Training"
test_data_path = root_path + "data/patData/2.Validation"
model = load_model(model_path)

### SET CUSTOM LABEL ###
data_type = "patent"
model, org_mappings, mappings, category = set_custom_label(model, data_type="patent")

inv_org_mappings = {}
for k, v in org_mappings.items():
    inv_org_mappings[v] = k

st.session_state["model"] = model
st.session_state["category"] = category
st.session_state["data_type"] = data_type
st.session_state["org_mappings"] = org_mappings
st.session_state["inv_org_mappings"] = inv_org_mappings
st.session_state["train_data_path"] = train_data_path
st.session_state["test_data_path"] = test_data_path
