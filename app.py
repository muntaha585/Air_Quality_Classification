import gradio as gr
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import load_model
import joblib

# Load saved models
mlp_sklearn = joblib.load("models/mlp_classifier.pkl")         
keras_model = load_model("models/keras_model.pkl")              
label_encoder = joblib.load("models/label_encoder.pkl")      
scaler = joblib.load("models/scaler.pkl")                    

def ensemble_predict(inputs):
    # Scale input
    X = scaler.transform([inputs])

    # Predict with both models
    sklearn_pred = mlp_sklearn.predict(X)[0]
    keras_probs = keras_model.predict(X)[0]
    keras_pred = np.argmax(keras_probs)

    # Ensemble: majority vote
    final_pred = sklearn_pred if sklearn_pred == keras_pred else keras_pred

    return label_encoder.inverse_transform([final_pred])[0]

# Input features (adjust based on your dataset)
feature_labels = [
    "CO (mg/m3)", "NO2 (ppb)", "O3 (ppb)", "PM2.5 (ug/m3)", "PM10 (ug/m3)"
]

def predict_interface(co, no2, o3, pm25, pm10, model_choice):
    inputs = [co, no2, o3, pm25, pm10]
    X = scaler.transform([inputs])

    if model_choice == "Sklearn MLP":
        pred = mlp_sklearn.predict(X)[0]
    elif model_choice == "Keras MLP":
        pred = np.argmax(keras_model.predict(X))
    else:
        pred = ensemble_predict(inputs)

    return label_encoder.inverse_transform([pred])[0]

# Gradio Interface
input_fields = [
    gr.Number(label=label) for label in feature_labels
] + [gr.Dropdown(["Sklearn MLP", "Keras MLP", "Ensemble"], label="Model")]

gr.Interface(
    fn=predict_interface,
    inputs=input_fields,
    outputs=gr.Textbox(label="Predicted Air Quality"),
    title="Air Quality Predictor",
    description="Select model and input environmental parameters to predict air quality"
).launch()
