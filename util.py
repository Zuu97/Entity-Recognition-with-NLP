import pandas as pd 
import numpy as np
from sklearn.utils import shuffle
from variables import*

def get_seperate_sentences(df, idx_i, idx_i_1):
    df_sentence = df.loc[idx_i:idx_i_1-1]
    words = df_sentence['Word'].values
    tags = np.array([label_dict[tag] for tag in df_sentence['Tag'].values])
    sentence = ' '.join(words.tolist())

    if max_length >= len(tags):
        padded_tags = np.append(tags, np.array([label_dict[oov_token]] * (max_length - len(tags))))
    else:
        padded_tags = tags[:max_length]
    return padded_tags, sentence

def get_sentences(df, not_nan_idxs):
    All_sentences = []
    All_tags = []
    for i, idx in enumerate(not_nan_idxs[:-1]):
        idx_i = idx 
        idx_i_1 = not_nan_idxs[i+1]
        padded_tags, sentence = get_seperate_sentences(df, idx_i, idx_i_1)

        All_sentences.append(sentence)
        All_tags.append(padded_tags)

    All_tags = np.array(All_tags)
    All_sentences = np.array(All_sentences)

    return All_tags, All_sentences

def load_data():
    df = pd.read_csv(data_path, encoding='ISO 8859-1')
    not_nan_idxs = np.nonzero(df['Sentence #'].notnull().values)[0]
    # max_length = np.diff(not_nan_idxs).max()
    All_tags, All_sentences = get_sentences(df, not_nan_idxs)
    All_tags, All_sentences = shuffle(All_tags, All_sentences)

    Ntest = int(len(All_sentences) * cutoff)
    Xtrain, Xtest = All_sentences[:-Ntest], All_sentences[-Ntest:]
    Ytrain, Ytest = All_tags[:-Ntest], All_tags[-Ntest:]
    return Xtrain, Xtest, Ytrain, Ytest