# 🧠 Casting Defect Detection using CNN

This project detects visual surface defects in casting products using a Convolutional Neural Network (CNN). It includes a Streamlit web app where users can upload grayscale casting images and receive real-time predictions with confidence scores.

## 🚀 Features
- Automatic defect detection from casting images
- Binary classification: OK ✅ or Defective ❌
- Real-time prediction using Streamlit
- Confidence score output
- Clean and minimal UI

## 🖼️ Sample Workflow
1. Upload a grayscale image (128x128 or auto-resized)
2. Model predicts class (OK or Defective)
3. Displays result + confidence bar

## 🛠️ Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- Streamlit
- NumPy, Matplotlib, Seaborn

## 📁 Project Structure
VISUAL DETECTION/
├── 01_data_preprocessing.ipynb
├── 02_model_training.ipynb
├── app.py
├── casting_defect_model.h5
└── README.md


## 📷 Dataset
[Kaggle - Real-life Industrial Dataset of Casting Product](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product)

## 🧪 Results
- Validation Accuracy: ~98%
- Test Accuracy: ~97%
- Real-time prediction in under 2 seconds

## ✍️ Author
**Kuldeep Amreliya & Team**  
From SAL College of Engineering  
GitHub: [itskd2k69](https://github.com/itskd2k69)

---

## 💡 Future Improvements
- Add support for more defect types
- Host the Streamlit app online
- Integrate Grad-CAM for visual explainability

---

## 📦 Run the App
```bash
streamlit run app.py
