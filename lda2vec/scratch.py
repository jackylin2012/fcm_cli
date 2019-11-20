import importlib

from SLda2vec import SLda2vec

if __name__ == "__main__":
    dataset_class = importlib.import_module("dataset.%s" % "news_group")
    data_attr = dataset_class.load_data()
    l2v = SLda2vec(**data_attr)
    l2v.fit()