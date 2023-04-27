import os
import pandas as pd
def fetch_data(rootdir, data_type, train_yn=True, train_len=170):
    dataset_train = []
    dataset_test = []
    dataset_all = []

    if data_type == "news":
        list_ = os.listdir(rootdir)
        for cat in list_:
            files = os.listdir(rootdir + cat)
            for i,f in enumerate(files):
                fname = rootdir + cat + "/" + f
                file_ = open(fname, "r", encoding="utf8")
                strings = file_.read()
                if i<train_len:
                    dataset_train.append([strings, cat])
                else:
                    dataset_test.append([strings, cat])
                dataset_all.append(strings)
                file_.close()

        print(len(dataset_train), len(dataset_test))

        if train_yn:
            docs = [dt[0] for dt in dataset_train]
            y = [int(dt[1]) for dt in dataset_train]
        else:
            docs = [dt[0] for dt in dataset_test]
            y = [int(dt[1]) for dt in dataset_test]

    elif data_type == "patent":
        if train_yn:
            data = pd.read_pickle(os.path.join(rootdir, "train.pkl"))
        else:
            data = pd.read_pickle(os.path.join(rootdir, "test.pkl"))

        mask = data["claims"].str.len() >= 1
        data = data.loc[mask]
        docs = data["claims"].astype("str").tolist()
        y = data["y"].astype("int").tolist()

    return docs, y