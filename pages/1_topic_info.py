import streamlit as st
from newsrec.tokenizer import CustomTokenizer
st.set_page_config(page_title="주제 및 주제 관련 키워드")
st.title("주제 추천 솔루션")

### LIST TOPIC ###
st.subheader("주제 및 주제 관련 키워드")
if "model" in st.session_state:
    model = st.session_state["model"]
    # topic_df = model.get_topic_info()
    # topic_df["Class"] = topic_df.Topic.map(mappings)
    # topic_df[["Name", "Class"]]
    st.plotly_chart(model.visualize_barchart(custom_labels=True))
