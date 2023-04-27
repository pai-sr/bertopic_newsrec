import streamlit as st
from topicrec.tokenizer import CustomTokenizer
st.set_page_config(page_title="입력 데이터에 대한 주제 추천")
st.title("주제 추천 솔루션")

### INPUT TEXT INFERENCE ###
st.subheader("주제 추천")
if "model" in st.session_state:

    import pandas as pd
    model = st.session_state["model"]
    category = st.session_state["category"]
    data_type = st.session_state["data_type"]
    inv_org_mappings = st.session_state["inv_org_mappings"]

    txt = st.text_area("주제를 추천받을 텍스트 입력", "텍스트를 입력하세요")

    if (txt != "") & (st.button("주제 추천")):
        topic, _ = model.transform(txt)
        if data_type == "news":
            st.write("result : ", category[topic[0]])
        elif data_type == "patent":
            st.write("result : ", category[inv_org_mappings[topic[0]]])

        embeddings = model._extract_embeddings(txt)
        umap_embeddings = model.umap_model.transform(embeddings)
        probs = model.hdbscan_model.predict_proba(umap_embeddings)
        df = pd.DataFrame([(c, '{:.3f}'.format(t)) for t, c in zip(probs[0], category)], columns=["topic", "probs"])
        st.write("prob_list : ", df.sort_values(by="probs", ascending=False).reset_index(drop=True))
    else:
        pass