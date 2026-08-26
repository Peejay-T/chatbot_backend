from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
from rag import answer_question, detect_emotion
from Title_generate import generate_chat_title
from datetime import datetime
from transformers import pipeline
import sqlite3
import websockets
import asyncio
from collections import Counter

app = FastAPI(title="University Chatbot API")

sentiment_model = pipeline( "sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment"
)

def init_db():
    conn = sqlite3.connect("chat_logs.db")
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS chat_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id TEXT,
        title TEXT,
        user_message TEXT,
        bot_response TEXT,
        timestamp TEXT
    )
    """)

    conn.commit()
    conn.close()

init_db()




app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    message: str
    history: list[dict] = []
    chat_id: Optional[str] = None
    title: Optional[str] = None

class TitleRequest(BaseModel):
    message: str


def save_log_db(chat_id, title, user_msg, bot_msg):
    conn = sqlite3.connect("chat_logs.db")
    c = conn.cursor()

    c.execute("""
    INSERT INTO chat_logs (chat_id, title, user_message, bot_response, timestamp)
    VALUES (?, ?, ?, ?, ?)
    """, (
        chat_id,
        title,
        user_msg,
        bot_msg,
        datetime.now().strftime("%d-%m-%Y %H:%M:%S") 
    ))

    conn.commit()
    conn.close()


@app.post("/chat")
async def chat(q: Query):
    try:
        answer = answer_question(q.message, q.history)

        final_title = q.title or generate_chat_title(q.message)

        save_log_db (
            chat_id=q.chat_id,
            title= final_title,
            user_msg=q.message, 
            bot_msg=answer
            )
        
         
        return {"answer": answer}

    except Exception as e:
        return {"error": str(e)}

@app.post("/generate-title")
async def generate_title(req: TitleRequest):
    print("🔥 TITLE ENDPOINT HIT", req.message)
    
    title = generate_chat_title(req.message)

    print(" GENERATED TITLE:", title)
    return {"title": title}

@app.get("/logs")
def get_logs():
    conn = sqlite3.connect("chat_logs.db")
    c = conn.cursor()

    c.execute("""
        SELECT 
            chat_id,
            title,
            user_message,
            bot_response,
            timestamp
        FROM chat_logs
        ORDER BY timestamp DESC
    """)

    rows = c.fetchall()
    conn.close()

    return rows 

def classify_inquiry(text):
    text = text.lower()

    if any(word in text for word in ["enroll", "admission", "register", "tuition"]):
        return "Enrollment Process"
    
    elif any(word in text for word in ["error", "bug", "not working", "disconnect", "failed"]):
        return "Technical Difficulties"
    
    elif any(word in text for word in ["schedule", "time", "when", "date"]):
        return "General Question"
    
    else:
        return "Other"

def analyze_sentiment(text):
    result = sentiment_model(text[:512])[0]
    label = result['label']  # e.g. "1 star", "5 stars"

    # Convert stars → sentiment
    if "1" in label or "2" in label:
        return "negative"
    elif "3" in label:
        return "neutral"
    else:
        return "positive"

@app.get("/analytics")
def analytics():
    conn = sqlite3.connect("chat_logs.db")
    cursor = conn.cursor()

    cursor.execute("SELECT title FROM chat_logs")
    rows = cursor.fetchall()

    result = {}

    for row in rows:
        text = row[0]
        if not text:
            continue

        category = classify_inquiry(text)
        sentiment = analyze_sentiment(text)

        if category not in result:
            result[category] = {"positive": 0, "negative": 0, "neutral":0}

        result[category][sentiment] += 1

    conn.close()  
    return result

@app.get("/top-questions")
def top_questions():
    conn = sqlite3.connect("chat_logs.db")
    cursor = conn.cursor()

    cursor.execute("SELECT title FROM chat_logs")
    rows = cursor.fetchall()

    # Extract text
    messages = [row[0].strip().lower() for row in rows if row[0]]

    # Count frequency
    counter = Counter(messages)

    # Get top 5
    top5 = counter.most_common(5)

    conn.close()

    # Format result
    return [
        {"question": q, "count": c}
        for q, c in top5
    ]

@app.get("/health")
async def health():
    return {"status": "ok"}

active_connections: list[WebSocket] = []

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/local")
async def local_ws(client_ws: WebSocket):
    await client_ws.accept()

    async with websockets.connect("ws://localhost:7000/ws/remote") as remote_ws:

        async def client_to_remote():
            while True:
                msg = await client_ws.receive_text()
                await remote_ws.send(msg)


        async def remote_to_client():
            while True:
                msg = await remote_ws.recv()
                await client_ws.send_text(msg)
      

        await asyncio.gather(
            client_to_remote(),
            remote_to_client()
        )
