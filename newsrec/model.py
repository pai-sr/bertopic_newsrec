from newsrec.dataset import fetch_data

from konlpy.tag import Mecab
from newsrec.tokenizer import CustomTokenizer
from sklearn.feature_extraction.text import CountVectorizer

from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.dimensionality import BaseDimensionalityReduction
from sklearn.linear_model import LogisticRegression


def load_basic_model():
    custom_tokenizer = CustomTokenizer(Mecab())
    vectorizer = CountVectorizer(tokenizer=custom_tokenizer, max_features=3000)

    empty_dimensionality_model = BaseDimensionalityReduction()
    clf = LogisticRegression()
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

    topic_model = BERTopic(
        embedding_model = "sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens",
        vectorizer_model = vectorizer,
        umap_model=empty_dimensionality_model,
        hdbscan_model=clf,
        ctfidf_model=ctfidf_model,
        calculate_probabilities=True
    )

    return topic_model

def fit_model(save_path, root_path):
    model = load_basic_model()

    train_dt, test_dt = fetch_data(rootdir=root_path)
    docs = [dt[0] for dt in train_dt]
    y = [int(dt[1]) for dt in train_dt]
    test_docs = [dt[0] for dt in test_dt]
    y_true = [int(dt[1]) for dt in test_dt]
    category = ["정치", "경제", "사회", "생활/문화", "세계", "기술/IT", "연예", "스포츠"]

    model.fit(documents=docs, y=y)

    model.save(save_path)

def load_model(model_path):
    model = BERTopic.load(model_path)
    return model