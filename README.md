# 🌫️ Air Quality Classification Using Perceptron Models & Neural Networks
A machine learning project focused on classifying air quality using various neural architectures—from a basic perceptron to advanced MLPs and ensemble methods—along with clustering for data exploration and a deployed Gradio interface for real-time predictions.

## 🧠 Models Implemented
1. Hardcoded Perceptron
A simple single-layer model with fixed weights and a step function. It lacks learning ability but serves as a baseline for comparison.
🔧 Manual Weights | 🚫 No Training | ⚙️ Step Activation

2. MLP Classifier (Scikit-learn)
A single hidden-layer neural network trained using backpropagation. Provides solid performance on structured data.
🧪 One Hidden Layer (5 neurons) | 🔁 150 Iterations | ✅ Accuracy: 0.92

3. MLP (Keras + Adam + Dropout)
A deep neural network with adaptive learning (Adam) and Dropout regularization to prevent overfitting.
🔥 ReLU & Softmax Activations | 🔄 Dropout Layers | 📈 Accuracy: 0.94

4. Ensemble Model
Combines predictions from both MLPs using majority voting to enhance robustness and stability.
🤝 Combined Outputs | 📊 Accuracy: 0.96

## ⚙️ Preprocessing Steps
Exploration: Used pandas and seaborn for initial inspection and correlation heatmaps
Missing Values: Checked and handled with .isna().sum()
Outlier Removal: Applied Z-score (>|3|)
Feature Scaling: Used Min-Max normalization
Encoding: Label-encoded target air quality categories
Feature Selection: Selected features with correlation > 0.5

## 📊 Clustering & Visualization
Clustering: Applied KMeans to group similar air quality data
Visualization: Used PCA to reduce dimensions and plot clusters in 2D
Insight: Showed clear class boundaries and cluster separations

## 🌐 Gradio Web Interface
Interactive deployment using Gradio:
Inputs environmental features
Allows switching between models (Sklearn MLP, Keras MLP, Ensemble)
Outputs predicted air quality label
Easy-to-use interface with real-time predictions

## 📈 Model Comparison
Feature	Hardcoded Perceptron	MLP (Sklearn)	MLP (Keras)	Ensemble
Training	None (manual weights)	Backpropagation	Adam + Dropout	Voting
Complexity	Very Low	Moderate	High	High
Layers	1	1 Hidden	Multi Hidden	Combined
Accuracy	0.48	0.92	0.94	0.96
Adaptability	None	Moderate	High	High
Activation	Step	ReLU, Logistic	ReLU, Softmax	Mixed

## 🚀 Future Improvements
🔧 Hyperparameter tuning
🌲 Add tree-based models (e.g., Random Forest, XGBoost)
⏳ Time series prediction for trend analysis
🔌 REST API integration for broader deployment

## 🔖 Directory Structure
air-quality-classifier/
│
├── models/
│   ├── mlp_classifier.pkl
│   ├── keras_model.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
│
├── app.py
├── requirements.txt
├── README.md
└── data/
    └── air_quality_dataset.csv

## 🧰 Tech Stack
Languages: Python
Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, keras, tensorflow, gradio
Algorithms: Perceptron, MLP, Ensemble Voting, KMeans, PCA

##📎 License
This project is open source and available under the MIT License.

👩‍💻 Author
Muntaha Nishat
BS Artificial Intelligence
✉️ Email: muntahanishat555@gmail.com
