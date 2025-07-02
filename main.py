
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load("spam_model.pkl")

class Message(BaseModel):
    text: str

@app.post("/predict")
def predict_spam(msg: Message):
    prediction = model.predict([msg.text])[0]
    return {"label": "spam" if prediction == 1 else "ham"}
