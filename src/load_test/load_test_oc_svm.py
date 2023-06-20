import numpy as np
import pickle
import time
import os

# Get PID
pid = os.getpid()

# Print PID
print("PID:", pid)

# Load test data
data = np.genfromtxt('../../datasets/unsupervised/test_data_norm.csv', delimiter=',')

# Load model
with open('../../models/oc_svm.pkl', 'rb') as f:
    oc_svm = pickle.load(f)
    
# Perform inference on test data
for item in  data:
    input_value = item.reshape(-1,1)
    anomaly = oc_svm.predict(input_value)
    anomaly[anomaly == 1] = 0
    anomaly[anomaly == -1] = 1
    # Add a 1 second sleep
    time.sleep(1)