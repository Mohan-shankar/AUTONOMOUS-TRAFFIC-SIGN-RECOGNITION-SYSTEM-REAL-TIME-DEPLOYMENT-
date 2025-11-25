# AUTONOMOUS-TRAFFIC-SIGN-RECOGNITION-SYSTEM-REAL-TIME-DEPLOYMENT-
Autonomous Traffic Sign Recognition system built using TensorFlow, EfficientNetB0, and Streamlit. The model is trained on the GTSRB dataset to classify 43 types of traffic signs in real-time with 93% accuracy. Includes a fully interactive Streamlit UI where users can upload images and instantly view predictions, confidence scores, and probability charts.

# 🚦 Autonomous Traffic Sign Recognition (GTSRB Dataset)

This project is a deep learning–based traffic sign recognition system built using:

- TensorFlow / Keras  
- EfficientNetB0  
- Streamlit for UI  
- GTSRB (German Traffic Sign Recognition Benchmark) Dataset  

---

## 📌 Features

### ✔️ Trainable Deep Learning Model
- EfficientNetB0 architecture  
- Trained for 20 epochs + 5 epochs separately 
- Handles 43 traffic sign classes  

### ✔️ Interactive Web App (Streamlit)
- Upload any traffic sign image  
- Real-time prediction  
- Shows:
  - Predicted class name  
  - Confidence score  
  - Probability bar graph  

### ✔️ Modular Code Structure
- `src/` → ML utilities  
- `streamlit/` → App interface  
- `notebook/` → Training notebook  
- `models/` → Saved model  

---

## 🚀 How to Run the Streamlit App

```bash
cd streamlit
streamlit run app.py
