import streamlit as st
st.title("뉴스기사 주제 추천 솔루션")

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

### LIST TOPIC ###
st.subheader("뉴스 기사 주제와 주제별 키워드")
#topic_df = model.get_topic_info()
#topic_df["Class"] = topic_df.Topic.map(mappings)
#topic_df[["Name", "Class"]]

st.plotly_chart(model.visualize_barchart(custom_labels=True))

### INPUT TEXT INFERENCE ###
st.subheader("주제 추천")
import pandas as pd
txt = st.text_area("주제를 추천받을 뉴스기사 텍스트 입력", "텍스트를 입력하세요")

if (txt != "") & (st.button("주제 추천")):
    topic, _ = model.transform(txt)
    st.write("result : ", category[topic[0]])

    embeddings = model._extract_embeddings(txt)
    umap_embeddings = model.umap_model.transform(embeddings)
    probs = model.hdbscan_model.predict_proba(umap_embeddings)
    df = pd.DataFrame([(c, '{:.3f}'.format(t)) for t, c in zip(probs[0], category)], columns=["topic", "probs"])
    st.write("prob_list : ", df.sort_values(by="probs", ascending=False).reset_index(drop=True))
else:
    pass

### PERFORMANCE CHECK ###
st.subheader("본 모델의 성능")
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
