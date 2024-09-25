import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

# モデルの初期化
model_name_or_path = 'sentence-transformers/bert-base-nli-mean-tokens'
model = SentenceTransformer(model_name_or_path)

# データセットの読み込み (ファイルパスを修正)
df = pd.read_csv(r"C:\Users\kenno\OneDrive\デスクトップ\product\tabelog_reviews.csv")

# NaN を空文字列に置き換え、データ型を文字列に変換
df['Comment'] = df['Comment'].fillna('').astype(str)

# 口コミをエンコード
df['review_embeddings'] = list(model.encode(df['Comment'].tolist(), convert_to_numpy=True))

def recommend_places(user_input, df, model, top_n=1):
    # ユーザー入力をエンコード
    user_embedding = model.encode(user_input, convert_to_numpy=True)

    # 全レビュー埋め込みとユーザー埋め込みの類似度を計算
    embeddings = np.vstack(df['review_embeddings'].values)
    similarities = util.cos_sim(user_embedding, embeddings).numpy().flatten()

    df['similarity'] = similarities
    recommendations = df.sort_values(by='similarity', ascending=False).head(top_n)

    return recommendations[['Store Name', 'Comment', 'similarity']]
