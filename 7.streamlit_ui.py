import streamlit as st
from ollama import chat
import numpy as np
from rich import print
import json
import pickle
import torch
from FlagEmbedding import BGEM3FlagModel
from tqdm.notebook import tqdm
from scipy import sparse
from sklearn.preprocessing import normalize
import pickle
import numpy as np


def lexical_to_csr(lexical_weights, vocab_size):
    """
    Convert list[defaultdict] → csr_matrix of shape (num_docs, vocab_size)
    """
    data = []
    rows = []
    cols = []

    for i, word_dict in enumerate(lexical_weights):
        for token_id_str, weight in word_dict.items():
            token_id = int(token_id_str)
            data.append(float(weight))         # convert np.float16 → float
            rows.append(i)
            cols.append(token_id)

    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(len(lexical_weights), vocab_size))
    return matrix


def get_clean_chunk(chunk):
    items = []
    for doc_item in chunk.meta.doc_items:
        items.append({
            "label": doc_item.label,
            "page_number": doc_item.prov[0].page_no,
            "charspan": doc_item.prov[0].charspan
         })
                
    clean_chunk = {
        "meta": {"filename": chunk.meta.origin.filename, "headings": chunk.meta.headings},
        "text": chunk.text
    }
    if items:
        clean_chunk["meta"]["chunk_items"] = items

    return clean_chunk

def hybrid_search(query, top_k=5, alpha=0.5):
    """
    alpha = weight for dense similarity (0-1)
    """
    q = model.encode([query], return_dense=True, return_sparse=True)

    # ----- Dense -----
    q_dense = np.array(q["dense_vecs"], dtype=np.float32).reshape(1, -1)
    q_dense = normalize(q_dense, axis=1)
    dense_scores = (dense_vectors @ q_dense.T).squeeze()
    
    # ----- Sparse -----
    q_sparse = lexical_to_csr([q["lexical_weights"][0]], vocab_size)

    sparse_scores = (sparse_vectors @ q_sparse.T).toarray().squeeze()

    # ----- Hybrid Score -----
    scores = alpha * dense_scores + (1 - alpha) * sparse_scores

    # ----- Top-K -----
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(chunks[i], float(scores[i])) for i in top_idx]


def augment_prompt(query, related_chunks):
    context_texts = []
    for chunk, score in related_chunks:
        context_texts.append(str(get_clean_chunk(chunk)))

    context = "\n\n---\n\n".join(context_texts)

    prompt = f"""
    سؤال کاربر:
    {query}

    تو یک دستیار هوشمند برای پاسخ‌دهی به پرسش‌ها بر اساس اسناد بازیابی‌شده هستی.
    فقط و فقط بر اساس متن‌های داده‌شده پاسخ بده.

    اگر پاسخ در متن‌ها وجود نداشت، حتماً فقط بنویس:
    «پاسخی یافت نشد.»

    هیچ دانش بیرونی، حدسی یا استنباط فراتر از متن مجاز نیست.

    در پاسخ نهایی، منبع را با این فرمت ذکر کن:

    نام heading (اگر وجود دارد)
    شماره صفحه (page_number از JSON)

    متن‌های بازیابی‌شده (قالب json):
    {context}
    """
    return prompt



# read from checkpoint
with open('checkpoints/embeddings-bge-m3.pkl', 'rb') as file:
    embeddings = pickle.load(file)

dense_vectors = np.array(embeddings["dense_vecs"], dtype=np.float32)
sparse_vectors = embeddings["lexical_weights"]  # scipy.sparse.csr_matrix

with open('checkpoints/clean_chunks.pkl', 'rb') as file:
    chunks = pickle.load(file)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)

model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, devices=device)
vocab_size = model.tokenizer.vocab_size
sparse_vectors = lexical_to_csr(embeddings["lexical_weights"], vocab_size)

# ------------------------
# STREAMLIT UI
# ------------------------

st.set_page_config(page_title="Persian RAG Assistant", layout="wide")

st.title("🔍 دستیار هوشمند فارسی")

# Query Input
query = st.text_input("پرسش کاربر:", placeholder="مثال: شرایط عمومی احراز صلاحیت افراد امتیازآور")

top_k = st.slider("🔢 تعداد قطعات برای بازیابی:", 1, 10, 5)
alpha = st.slider("⚖️ وزن شباهت برداری vs واژگانی (alpha):", 0.0, 1.0, 0.7)

show_prompt = st.checkbox("📄 نمایش پرامپت ساخته‌شده")

if st.button("🚀 جستجو و پاسخ"):
    if not query.strip():
        st.warning("لطفاً یک پرسش وارد کنید.")
    else:
        with st.spinner("در حال بازیابی اسناد..."):
            related_chunks = hybrid_search(query, top_k=top_k, alpha=alpha)

        with st.spinner("در حال تولید پاسخ مدل..."):
            prompt = augment_prompt(query, related_chunks)
            response = chat(model="Gemma3:12b", messages=[{"role": "user", "content": prompt}])
            answer = response["message"]["content"]

        st.markdown("## ✅ پاسخ:")
        st.write(answer)

        if show_prompt:
            st.markdown("---")
            st.markdown("### 📄 پرامپت ساخته‌شده")
            st.code(prompt, language="markdown")

        st.markdown("---")
        st.markdown("### 📚 بخش‌های بازیابی‌شده")
        for i, (chunk, score) in enumerate(related_chunks, start=1):
            st.markdown(f"**[{i}] امتیاز:** `{score:.4f}`")
            st.json(chunk)   # Assuming chunk is JSON-like
            st.markdown("---")
