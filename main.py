from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

app = FastAPI()

# ✅ Add CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict to your frontend domain if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("spam_model.pkl")

class Message(BaseModel):
    text: str

@app.post("/predict")
def predict_spam(msg: Message):
    prediction = model.predict([msg.text])[0]
    return {"label": "spam" if prediction == 1 else "ham"}
