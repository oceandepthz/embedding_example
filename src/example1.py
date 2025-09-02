import os

import numpy as np
from openai import OpenAI
import google.generativeai as genai
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

# OpenAIクライアントの初期化
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Google Geminiクライアントの初期化
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_client = genai


def get_embedding(text, model_name):
    if isinstance(text, str):
        text = [text]
        
    if "text-embedding" in model_name:
        response = openai_client.embeddings.create(
            model=model_name,
            input=text
        )
        return [np.array(d.embedding) for d in response.data]
    elif "gemini-embedding" in model_name: 
        response = gemini_client.embed_content(
            model=model_name,
            content=text
        )
        return [np.array(e) for e in response['embedding']]
    else:
        raise ValueError("Unsupported model.")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

evaluation_data = {
    "similar": [
        ["AIは人間の仕事を奪うのではなく、協働するパートナーとなる。", "人工知能は仕事の効率を上げ、人間との協力を可能にする存在だ。"],
        ["今日の天気は晴れです。", "本日は快晴なり。"],
        ["このレストランの料理はとても美味しい。", "ここの食事は最高だね。"]
    ],
    "unrelated": [
        ["今日の夕食はカレーにしよう。", "火星探査機の最新データが公開された。"],
        ["猫が日向ぼっこをしている。", "円周率の計算記録が更新された。"],
        ["新しいイヤホンを買った。", "日本の総人口は減少傾向にある。"]
    ],
    "multilingual": [
        ["新しいスマートフォンの発表が期待されている。", "The announcement of a new smartphone is highly anticipated."],
        ["これはペンです。", "This is a pen."],
        ["私はこの映画が大好きです。", "I love this movie."]
    ]
}

models_to_test = {
    "OpenAI-small": "text-embedding-3-small",
    "OpenAI-large": "text-embedding-3-large",
    "Gemini": "gemini-embedding-001"
}

results_list = []

for model_key, model_name in models_to_test.items():
    print(f"--- {model_key} モデルをテスト中 ---")
    
    # 各シナリオで評価
    for scenario, pairs in evaluation_data.items():
        scores = []
        for pair in pairs:
            # 2つのテキストを一度にエンベディング（API効率化）
            embeddings = get_embedding(pair, model_name)
            emb1 = embeddings[0]
            emb2 = embeddings[1]
            
            score = cosine_similarity(emb1, emb2)
            scores.append(score)
            print(str(pair))
            print(score)
        
        # 平均スコアを計算
        avg_score = np.mean(scores)
        print(f"{scenario}ペアの平均類似度: {avg_score:.4f}")
        
        results_list.append({
            "model": model_key,
            "scenario": scenario,
            "avg_score": avg_score
        })
    print("-" * 20)

# --- 改善点3: 結果をDataFrameで見やすく表示 ---
df_results = pd.DataFrame(results_list)
print("\n--- 総合結果 ---")
# モデルごと、シナリオごとに結果をピボットして見やすくする
pivot_table = df_results.pivot(index="model", columns="scenario", values="avg_score")
print(pivot_table)
