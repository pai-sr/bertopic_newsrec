from topicrec.dataset import fetch_data

from konlpy.tag import Mecab
from topicrec.tokenizer import CustomTokenizer
from sklearn.feature_extraction.text import CountVectorizer

from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.dimensionality import BaseDimensionalityReduction
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer

def build_model():
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
        calculate_probabilities=True,
        verbose=True
    )

    return topic_model

def fit_model(save_path, root_path, data_type):
    model = build_model()
    docs, y = fetch_data(rootdir=root_path, data_type=data_type)
    model.fit(documents=docs, y=y)

    model.save(save_path)

def load_model(model_path):
    try:
        model = BERTopic.load(model_path)
    except RuntimeError as e:
        model_path = model_path + "_noembedding"
        embedding_model = SentenceTransformer("sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens")
        model = BERTopic.load(model_path, embedding_model=embedding_model)
    return model