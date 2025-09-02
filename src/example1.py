import os
import numpy as np
from openai import OpenAI
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# OpenAIクライアントの初期化
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Google Geminiクライアントの初期化
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_client = genai

def get_embedding(text, model_name):
    if "text-embedding" in model_name:
        response = openai_client.embeddings.create(
            model=model_name,
            input=text
        )
        return np.array(response.data[0].embedding)
    elif "gemini-embedding" in model_name:
        response = gemini_client.embed_content(
            model=model_name,
            content=text
        )
        return np.array(response['embedding'])
    else:
        raise ValueError("Unsupported model.")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

# 実験データの定義
similar_pair = ["AIは人間の仕事を奪うのではなく、協働するパートナーとなる。", "人工知能は仕事の効率を上げ、人間との協力を可能にする存在だ。"]
unrelated_pair = ["今日の夕食はカレーにしよう。", "火星探査機の最新データが公開された。"]
multilingual_pair = ["新しいスマートフォンの発表が期待されている。", "The announcement of a new smartphone is highly anticipated."]

models_to_test = {
    "OpenAI-small": "text-embedding-3-small",
    "OpenAI-large": "text-embedding-3-large",
    "Gemini": "gemini-embedding-001"
}

results = {}

for model_key, model_name in models_to_test.items():
    print(f"--- {model_key} モデルをテスト中 ---")

    # 似ている文章ペア
    emb_A = get_embedding(similar_pair[0], model_name)
    emb_B = get_embedding(similar_pair[1], model_name)
    sim_score = cosine_similarity(emb_A, emb_B)
    print("--- 似ている文章ペア")
    print(str(similar_pair))
    print(f"似ているペアの類似度: {sim_score:.4f}")
    results[f"{model_key}_similar"] = sim_score

    # 無関係な文章ペア
    emb_C = get_embedding(unrelated_pair[0], model_name)
    emb_D = get_embedding(unrelated_pair[1], model_name)
    unrelated_score = cosine_similarity(emb_C, emb_D)
    print("--- 無関係な文章ペア")
    print(str(unrelated_pair))
    print(f"無関係なペアの類似度: {unrelated_score:.4f}")
    results[f"{model_key}_unrelated"] = unrelated_score

    # 多言語ペア
    emb_E = get_embedding(multilingual_pair[0], model_name)
    emb_F = get_embedding(multilingual_pair[1], model_name)
    multilingual_score = cosine_similarity(emb_E, emb_F)
    print("--- 多言語ペア")
    print(str(multilingual_pair))
    print(f"多言語ペアの類似度: {multilingual_score:.4f}\n")
    results[f"{model_key}_multilingual"] = multilingual_score
