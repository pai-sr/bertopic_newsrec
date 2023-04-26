import streamlit as st
from newsrec.tokenizer import CustomTokenizer
st.set_page_config(page_title="입력 데이터에 대한 뉴스 주제 추천")
st.title("뉴스기사 주제 추천 솔루션")

### PERFORMANCE CHECK ###
st.subheader("본 모델의 성능")
if "model" in st.session_state:
    model = st.session_state["model"]
    data_type = st.session_state["data_type"]
    test_data_path = st.session_state["test_data_path"]
    inv_org_mappings = st.session_state["inv_org_mappings"]

if st.button("성능 확인"):
    ### LOAD DATA ###
    from newsrec.dataset import fetch_data
    @st.cache_data
    def wrap_fetch_data(rootdir, data_type, train_yn=False):
        return fetch_data(rootdir, data_type, train_yn=False)

    test_docs, y_true = wrap_fetch_data(test_data_path, data_type, False)

    ### TEST DATA INFERENCE ###
    from tqdm import tqdm
    from sklearn.metrics import accuracy_score
    y_preds = []
    with st.spinner("Waiting. . ."):
        for dt_test in tqdm(test_docs):
            if data_type == "news":
                y_pred, _ = model.transform(dt_test)
                y_pred = y_pred[0]
            elif data_type == "patent":
                y_pred, _ = model.transform(dt_test)
                y_pred = inv_org_mappings[y_pred[0]]
            y_preds.append(y_pred)

    acc = accuracy_score(y_true, y_preds)
    st.write("accuracy : ", acc)