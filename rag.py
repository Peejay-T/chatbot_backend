import openai
from transformers import pipeline
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

qdrant = QdrantClient("localhost", port=6333)
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed(text):
    return model.encode(text).tolist()

def retrieve(query):
    vec = embed(query)
    hits = qdrant.query_points(collection_name="PLMUN_Chatbot_Knowledge", query=vec, limit=3)
    points = hits.points
    snippets = [p.payload.get('text') for p in points]
    return [s for s in snippets if s]

def detect_emotion(text):
    res = emotion_model(text)[0]
    return res["label"], res["score"]

def answer_question(user_query, history):
    emotion, score = detect_emotion(user_query)
    context = "\n".join(retrieve(user_query))

    prompt = f"""You are UniAssist, an empathetic context-aware university assistant chatbot. Use ONLY the Context below to answer the question. \n\nCONTEXT:\n{context}\n\nAnswer concisely 

User emotion detected: {emotion}

Adapt tone:
- If stressed/upset/angry, show understanding and apologize
- If confused, explain clearly step-by-step 
- If happy/excited, respond positively and greet the student
 

Relevant Context:
{context}

User Question:
{user_query}
"""

    res = openai.chat.completions.create(
        model="gpt-5.2",
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content