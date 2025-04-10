import requests
import json

# Define API endpoint
url = "http://34.45.66.85:5000/invocations"

# Prepare input data
data = {
    "instances": [[1200], [2000]]      # Input data
}

# Send POST request
response = requests.post(url, json=data)

# Print response
print("Predictions:", response.json())