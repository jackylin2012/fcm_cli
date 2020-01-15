import os
import pickle
import re
import string

import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from dataset.base_dataset import BaseDataset, encode_documents

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.abspath(os.path.join(CURRENT_DIR, os.pardir, os.pardir)), "data", "prosper_loan")
MIN_DF = 0.01
MAX_DF = 0.8
TEST_RATIO = 0.15


TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)

trans_table = {ord(c): None for c in string.punctuation + string.digits}
stemmer = nltk.SnowballStemmer("english")

def tokenize(text):
        tokens = [word for word in nltk.word_tokenize(text.translate(trans_table)) if len(word) >= 3]
        stems = [stemmer.stem(item) for item in tokens]
        return stems


class WordsSweatDataset(BaseDataset):
    def __init__(self):
        print("Reading data...")
        x_df = pd.read_csv(os.path.join(DATA_DIR, "loanfatetable.csv"))
        x_df['CreationDate'] = pd.to_datetime(x_df['CreationDate'])
        x_df = x_df[(x_df.CreationDate >= pd.to_datetime('2007-04-01')) & (x_df.CreationDate <= pd.to_datetime('2008-10-01'))]
        x_df = x_df[['Key', 'AmountRequested', 'CreditGrade', 'DebtToIncome', 'IsBorrowerHomeowner', 'LenderRate', 'LoanStatus']]

        listing_df = pd.read_csv(os.path.join(DATA_DIR, "Listings.CSV"), engine='python')
        listing_df['CreationDate'] = pd.to_datetime(listing_df['CreationDate'])
        listing_df = listing_df[(listing_df.CreationDate >= pd.to_datetime('2007-04-01T00:00:00')) & (listing_df.CreationDate < pd.to_datetime('2008-10-02T00:00:00'))]
        listing_df = listing_df[['Key', 'Description']]
        x_df = pd.merge(x_df, listing_df, on='Key')
        x_df = x_df[x_df['LoanStatus'].isin(['Charge-off', 'Defaulted (Bankruptcy)', 'Defaulted (Delinquency)', 'Paid', 'Defaulted (PaidInFull)', 'Defaulted (SettledInFull)'])]
        labels = x_df['LoanStatus'].isin(['Paid', 'Defaulted (PaidInFull)', 'Defaulted (SettledInFull)']).astype(int).to_numpy()
        x_df = pd.concat([x_df, pd.get_dummies(x_df['CreditGrade'], prefix='CreditGrade')], axis=1)
        x_df = x_df.drop(columns="CreditGrade")
        x_df['DebtToIncomeMissing'] = x_df['DebtToIncome'].isna().astype(int)
        x_df['DebtToIncome'] = x_df['DebtToIncome'].fillna(0)
        x_df['Description'] = x_df['Description'].fillna('')
        documents = x_df['Description'].str.replace('(<[^>]+>)|([^\x00-\x7F]+)', ' ', regex=True)\
            .str.replace('&nbsp;', ' ', regex=False).str.replace('\n', ' ', regex=False).str.replace('\t', ' ', regex=False)
        x_df.to_csv(os.path.join(DATA_DIR, "prosper_loan.csv"))
        x_df = x_df.drop(columns='LoanStatus')
        x_df = x_df.drop(columns='Description')
        # explanatory vars
        expvars = x_df.to_numpy()

        self.doc_train, self.doc_test, self.y_train, self.y_test, self.expvars_train, self.expvars_test = \
            train_test_split(documents, labels, expvars, test_size=TEST_RATIO)

    def load_data(self, params):
        window_size = params["window_size"]  # context window size
        vocab_size = params["vocab_size"]  # max vocabulary size
        min_df = params.get("min_df", MIN_DF)  # min document frequency of vocabulary, defaults to MIN_DF
        max_df = params.get("max_df", MAX_DF)  # max document frequency of vocabulary, defaults to MAX_DF
        file_name = os.path.join(DATA_DIR, "prosper_%d.pkl" % window_size)
        if os.path.exists(file_name):
            WordsSweatDataset.data = pickle.load(open(file_name, "rb"))
            return WordsSweatDataset.data

        vectorizer = CountVectorizer(tokenizer=tokenize, stop_words='english',min_df=min_df, max_df=max_df, max_features=vocab_size)
        X_train, y_train, X_test, wordcounts_train, doc_lens, vocab, doc_windows_train, _ = \
            encode_documents(vectorizer, window_size, self.doc_train, self.y_train, self.doc_test, self.expvars_train)
        data = {
            "doc_windows": doc_windows_train,
            "word_counts": wordcounts_train,
            "doc_lens": doc_lens,
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": self.y_test,
            "vocab": vocab,
            "expvars_train": self.expvars_train,
            "expvars_test": self.expvars_test
        }
        pickle.dump(data, open(file_name, "wb"))
        return data
