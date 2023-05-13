import datetime
import tensorflow as tf
import numpy as np
import joblib
from typing import List
from fastapi import FastAPI, Body, WebSocket
from uvicorn import Config
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

class Anomaly(BaseModel):
    value: float
    anomaly: int

class pointData(BaseModel):
    value: float

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

with open("static/LSTM_model.json") as f:
        model_json = f.read()
f.close()

model = tf.keras.models.model_from_json(model_json)
model.load_weights("static/LSTM_model.h5")

scaler = joblib.load("static/scaler.save") 

mae = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)

THRESHOLD = 2.6519394

MODE = "standalone" # cloud => Cloud mode, standalone => Standalone mode

last_observations = []

def predict_anomaly(losses, threshold = THRESHOLD):
    anomalies_array = []
    for losses_timestep in losses:
        anomaly = []
        for loss in losses_timestep:
            if loss > threshold:
                anomaly.append(1)
            else:
                anomaly.append(0)

        anomalies_array.append(anomaly)
    return np.array(anomalies_array)

class WebSocketManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)
    
    async def send_anomaly_json(self, data):
        for connection in self.active_connections:
            await connection.send_json(data)

websocket_manager = WebSocketManager()

@app.get("/")
async def root():
    with open("static/index.html") as f:
        html = f.read()
    f.close()
    return HTMLResponse(html)

@app.get("/get_mode/")
async def get_mode():
    print(MODE)
    return MODE

@app.post("/anomaly/")
async def receive_anomaly(item: Anomaly):
    print(item)
    data = {
        "label": datetime.datetime.now().isoformat(),
        "value": item.value,
        "anomaly": item.anomaly
    }
    await websocket_manager.send_anomaly_json(data)
    return "OK"

@app.post("/predict/")
async def predict(item: pointData):
    if len(last_observations) > 9:
        last_observations.pop(0)
    last_observations.append(item.value)
    if len(last_observations) == 10:
        norm_data = scaler.transform(np.array(last_observations).reshape(-1, 1))
        entrada = np.array(norm_data).reshape(-1,10,1)
        prediccion = model.predict(entrada)
        losses = mae(entrada, prediccion).numpy()
        anomalies = predict_anomaly(losses)[0]
        print(anomalies[-1])
        data = {
            "label": datetime.datetime.now().isoformat(),
            "value": item.value,
            "anomaly": int(anomalies[-1])
        }
        await websocket_manager.send_anomaly_json(data)
        return "OK"
    return "Needs more data"

@app.post("/predict_text/")
async def predict_text(body_data: str = Body(...)):
    print("Body data: ")
    print(body_data)
    return "OK"

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket_manager.connect(websocket)
    try:
        while True:
            message = await websocket.receive_text()
            await websocket_manager.broadcast(message)
    except Exception:
        websocket_manager.disconnect(websocket)

