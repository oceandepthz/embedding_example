import os

import numpy as np
from openai import OpenAI
import google.generativeai as genai
from dotenv import load_dotenv
import pandas as pd

# .envファイルから環境変数を読み込む
load_dotenv()

# OpenAIクライアントの初期化
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Google Geminiクライアントの初期化
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_client = genai


def get_embedding(text, model_name):
    """テキストから埋め込みベクトルを取得する関数"""
    # 入力が単一の文字列の場合、リストに変換
    if isinstance(text, str):
        text = [text]
        
    if "text-embedding" in model_name:
        response = openai_client.embeddings.create(
            model=model_name,
            input=text
        )
        # レスポンスからnumpy配列のリストを返す
        return [np.array(d.embedding) for d in response.data]
    elif "gemini-embedding" in model_name: 
        response = gemini_client.embed_content(
            model=model_name,
            content=text
        )
        # レスポンスからnumpy配列のリストを返す
        return [np.array(e) for e in response['embedding']]
    else:
        raise ValueError("Unsupported model.")

def cosine_similarity(vec1, vec2):
    """2つのベクトルのコサイン類似度を計算する関数"""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    # ゼロ除算を避ける
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return dot_product / (norm_vec1 * norm_vec2)

# --- ここからが新しい検証コード ---

# 検証に使用するモデル
models_to_test = {
    "OpenAI-small": "text-embedding-3-small",
    "OpenAI-large": "text-embedding-3-large",
    "Gemini": "gemini-embedding-001"
}

# 検証用のテキストリスト
texts_to_verify = [
    "AIは人間の仕事を奪うのではなく、協働するパートナーとなる。",
    "今日の天気は晴れです。",
    "This is a pen.",
    "The announcement of a new smartphone is highly anticipated."
]

print("\n--- 同一テキスト・同一モデルでのベクトル再現性検証 ---")

# 結果を格納するためのリスト
verification_results = []

# 各モデルで検証を実行
for model_key, model_name in models_to_test.items():
    print(f"\n--- モデル: {model_key} ({model_name}) を検証中 ---")
    
    for text in texts_to_verify:
        print(f"テキスト: \"{text[:30]}...\"")
        
        # 同じテキストで2回ベクトルを生成する
        # get_embeddingはリストを返すので、最初の要素([0])を取得する
        vec1 = get_embedding(text, model_name)[0]
        vec2 = get_embedding(text, model_name)[0]
        
        # 2つのベクトルを比較
        # 1. 完全一致の確認
        are_equal = np.array_equal(vec1, vec2)
        
        # 2. コサイン類似度の計算
        similarity = cosine_similarity(vec1, vec2)
        
        # 3. ユークリッド距離（ベクトル間の差の大きさ）の計算
        euclidean_distance = np.linalg.norm(vec1 - vec2)
        
        # 個別の結果を表示
        print(f"  ベクトルは完全に一致するか？: {are_equal}")
        print(f"  コサイン類似度: {similarity:.10f}")
        print(f"  ユークリッド距離: {euclidean_distance:.10e}") # 指数表記で微小な差も表示
        
        # 総合結果用のリストにデータを追加
        verification_results.append({
            "model": model_key,
            "text": text,
            "are_equal": are_equal,
            "cosine_similarity": similarity,
            "euclidean_distance": euclidean_distance,
        })

# --- 検証結果をDataFrameでまとめて表示 ---
print("\n" + "="*50)
print("--- 総合結果 ---")
print("="*50)

df_verification = pd.DataFrame(verification_results)

# テキストが長い場合、表示を短縮する
df_verification['text_short'] = df_verification['text'].str[:20] + '...'
df_verification = df_verification.drop(columns=['text'])

# 表示する列の順番を整理
column_order = [
    "model", "text_short", "are_equal", "cosine_similarity", 
    "euclidean_distance"
]
df_verification = df_verification[column_order]

# 浮動小数点数の表示形式を設定
pd.set_option('display.float_format', '{:.8f}'.format)
pd.set_option('display.max_rows', None) # 全ての行を表示

print(df_verification)
