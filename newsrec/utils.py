import json

with open("patData/label_mapping.json", "r", encoding="utf-8") as f:
    label_mapping = json.load(f)

news_category = ["정치", "경제", "사회", "생활/문화", "세계", "기술/IT", "연예", "스포츠"]
patent_category = list(label_mapping.values())

def set_custom_label(model, data_type):
    org_mappings = model.topic_mapper_.get_mappings()
    if data_type == "news":
        mappings = {value: news_category[key] for key, value in org_mappings.items()}
        model.set_topic_labels(mappings)
        category = news_category
    elif data_type == "patent":
        mappings = {value: patent_category[key] for key, value in org_mappings.items()}
        model.set_topic_labels(mappings)
        category = patent_category
    else:
        raise NotImplementedError
    return model, org_mappings, mappings, category