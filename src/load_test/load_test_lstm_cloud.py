import numpy as np
import tensorflow as tf
import time
import os

# Get PID
pid = os.getpid()

# Print PID
print("PID:", pid)

# Load test data
data = np.genfromtxt('../../datasets/semisupervised/test_data_norm.csv', delimiter=',')

# Threshold
THRESHOLD = 0.6844909

# Timesteps
t = 6

# Function that receives the loss array and the THRESHOLD and performs anomaly detection
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

# Inicialize Mean Absolute Error metric
mae = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)

# Load model architecture from JSON file
with open('../../models/LSTM_model.json', 'r') as json_file:
    model_json = json_file.read()

# Create the model from the loaded architecture
model = tf.keras.models.model_from_json(model_json)

# Load model weights from H5 file
model.load_weights('../../models/LSTM_model.h5')

# Perform inference on test data
for item in  data:
    input_value = item.reshape(1, t, 1).astype(np.float32)
    prediction = model.predict(input_value, verbose = 0)
    losses = mae(input_value, prediction).numpy()
    anomalies = predict_anomaly(losses)[0]
    # Add a 1 second sleep
    time.sleep(1)

