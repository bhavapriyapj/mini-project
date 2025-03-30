from flask import Flask, request, jsonify, send_from_directory
import torch
import numpy as np
from torch_geometric.data import Data
from model import GNN
import os
import joblib
import random
import torch.nn.functional as F  # Importing functional utilities

app = Flask(__name__)

model_path = "C:\\Users\\USER\\miniproject\\sample\\backend\\gnn_model.pth"
scaler_path = "C:\\Users\\USER\\miniproject\\sample\\backend\\scaler.pkl"

# Check if model and scaler exist
if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    print("Error: Missing model or scaler file.")
    exit(1)

# Initialize model and load parameters
num_features = 11
num_classes = 2
model = GNN(num_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()
scaler = joblib.load(scaler_path)

# Temporary storage for OTPs
otp_storage = {}

@app.route('/')
def index():
    return send_from_directory('../frontend', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        transaction = np.array(data["features"], dtype=np.float32)

        # Validate input length
        if len(transaction) != num_features:
            return jsonify({"error": f"Expected {num_features} features, got {len(transaction)}"})

        # Validate value range
        if any(x < -1e6 or x > 1e6 for x in transaction):
            return jsonify({"error": "Feature values out of expected range"})

        print("Received features:", transaction)
        
        # Scale transaction input
        transaction = scaler.transform(transaction.reshape(1, -1))
        print("Scaled features:", transaction)

        # Convert input to tensor
        x = torch.tensor(transaction, dtype=torch.float)  # Shape: [1, 11]
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)  # Self-loop

        # Create PyTorch Geometric data object
        test_data = Data(x=x, edge_index=edge_index)

        # Model inference
        with torch.no_grad():
            output = model(test_data)
            probabilities = F.softmax(output, dim=1).numpy()  # Compute softmax probabilities
            _, prediction = output.max(dim=1)
            fraud_probability = probabilities[0][1]  # Extract fraud probability

        print("Model output:", output.numpy())
        print("Prediction:", prediction.item())
        print("Fraud probability:", fraud_probability)

        risk_score = fraud_probability * 100  # Convert to percentage
        print(f"Risk Score: {risk_score:.2f}%")

        # Ensure fraud_probability and risk_score are converted to native Python floats
        response_data = {
            "fraud_prediction": int(prediction.item()),
            "confidence": float(fraud_probability),  # Convert to native Python float
            "risk_score": float(risk_score)  # Convert to native Python float
        }

        # Fraud detection logic
        if fraud_probability > 0.7:
            otp = random.randint(1000, 9999)  # Generate a 4-digit OTP
            otp_storage[data["user_id"]] = otp
            print(f"OTP sent: {otp}")
            response_data.update({
                "message": "🚨 High risk detected! OTP sent for confirmation.",
                "otp": otp
            })
        elif fraud_probability > 0.3:
            response_data["message"] = "⚠️ Medium risk detected. Please verify your transaction."
        else:
            response_data["message"] = "✅ Low risk. Transaction successful."

        return jsonify(response_data)

    except Exception as e:
        return jsonify({"status": "API is running", "error": str(e)})

@app.route('/verify_otp', methods=['POST'])
def verify_otp():
    try:
        data = request.json
        user_id = data.get("user_id")
        otp = int(data.get("otp"))  # Ensure OTP is treated as an integer

        # Validate OTP existence
        if user_id not in otp_storage:
            return jsonify({"error": "No OTP found for this user."})

        stored_otp = otp_storage[user_id]

        if otp == stored_otp:
            del otp_storage[user_id]  # Clear OTP after verification
            return jsonify({"message": "✅ OTP verified successfully. Transaction completed."})
        else:
            return jsonify({"error": "❌ Invalid OTP. Please try again."})

    except Exception as e:
        return jsonify({"status": "API is running", "error": str(e)})

@app.route('/status', methods=['GET'])
def status():
    return jsonify({"status": "API is running"})

if __name__ == '__main__':
    app.run(debug=True)
