import os


dataset_train = []
dataset_test = []
dataset_all = []

def fetch_data(rootdir="newsData/", train_len=170):
    list = os.listdir(rootdir)
    for cat in list:
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
    return dataset_train, dataset_test