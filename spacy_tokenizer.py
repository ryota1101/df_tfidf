from spacy.lang.ja import Japanese
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def spacy_tokenizer(text):
    nlp = Japanese()
    doc = nlp(text)
    tokens = [token.text for token in doc]
    return tokens

def spacy_extract(text):
    nlp = Japanese()
    doc = nlp(text)
    pos_list = ["NOUN", "VERB", "ADJ"]
    words = [token.lemma_ for token in doc if token.pos_ in pos_list]
    
    return ' '.join(words)

#df全体に対してmecav_tokenizerを適用し、形態素解析を行なったリストを返す関数
def make_docs(df,column_number):
    docs=[]
    for i in range(len(df)):
        text = df.iloc[i,column_number]
        docs.append(spacy_extract(text))
    return docs

def tfidf_prune_sentence(df, target_col_num, threshold=0.2):
    # テキストを形態素解析してdocsリストを作成
    docs = make_docs(df, target_col_num)
    
    # TF-IDFベクトライザの初期化とフィッティング
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(docs)
    
    # TF-IDF値をデータフレームに格納
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

    # TF-IDF値が閾値未満の行を削除
    rows_to_keep = (tfidf_df > threshold).any(axis=1)
    filtered_df = df[rows_to_keep]
    
    return filtered_df

if __name__ == "__main__":
    data = {
        'text': ["梅干し食べたい","梅干し食べたい", "梅干し食べたい", "梅干し食べたい","カレー食べたい","梅干し結構好き"],
        'num': ["0","1", "2", "3","4","5"]
    }
    df = pd.DataFrame(data)
    
    df = tfidf_prune_sentence(df, df.columns.get_loc('text'),0.8)

    print(df)