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
THRESHOLD = 0.102877855

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

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path='../../models/model_autoencoder.tflite')
interpreter.allocate_tensors()

# Get model input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Perform inference on test data
for item in  data:
    input_value = item.reshape(1, t, 1).astype(np.float32)
    # Configure the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_value)
    # Make the inference
    interpreter.invoke()
    # Get results
    output_data = interpreter.get_tensor(output_details[0]['index'])
    losses = mae(input_value, output_data).numpy()
    anomalies = predict_anomaly(losses)[0]
    # Add a 1 second sleep
    time.sleep(1)
