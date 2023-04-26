import streamlit as st
from newsrec.tokenizer import CustomTokenizer
st.set_page_config(page_title="입력 데이터에 대한 뉴스 주제 추천")
st.title("뉴스기사 주제 추천 솔루션")

### PERFORMANCE CHECK ###
st.subheader("본 모델의 성능")
if "model" in st.session_state:
    model = st.session_state["model"]

if st.button("성능 확인"):
    ### LOAD DATA ###
    from newsrec.dataset import fetch_data
    @st.cache_data
    def wrap_fetch_data():
        return fetch_data()
    train_dt, test_dt = wrap_fetch_data()
    docs = [dt[0] for dt in train_dt]
    y = [int(dt[1]) for dt in train_dt]
    test_docs = [dt[0] for dt in test_dt]
    y_true = [int(dt[1]) for dt in test_dt]

    ### TEST DATA INFERENCE ###
    from tqdm import tqdm
    from sklearn.metrics import accuracy_score
    y_preds = []
    with st.spinner("Waiting. . ."):
        for dt_test in tqdm(test_docs):
            y_pred, _ = model.transform(dt_test)
            y_preds.append(y_pred[0])
    acc = accuracy_score(y_true, y_preds)
    st.write("accuracy : ", acc)