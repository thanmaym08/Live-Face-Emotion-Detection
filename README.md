# Live Face Emotion Detection 🎭  

A real-time face emotion detection system using **MobileNetV2**, trained on a custom dataset. The model detects emotions and displays them on screen.  

---

## 🚀 Features  
✅ Real-time face emotion detection using **OpenCV** and **MobileNetV2**  
✅ Custom-trained model for improved accuracy  
✅ Supports multiple emotions (Happy, Sad, Angry, etc.)  
✅ Uses **Python, TensorFlow, OpenCV, and NumPy**  
✅ Lightweight and optimized for real-time performance  

---

## 📂 Project Structure  

📦 Live Face Emotion Detection

┣ 📂 dataset/ # Custom dataset for training

┣ 📂 models/ # Trained model files (.h5)

┣ 📂 src/

┃ ┣ 📜 real_time_emotion.py # Real-time emotion detection script

┃ ┣ 📜 train_model.py # Training script for MobileNetV2

┃ ┣ 📜 requirements.txt # Required dependencies

┃ ┗ 📜 README.md

┣ 📂 venv/ # Virtual environment (ignored in Git)

┗ 📜 .gitignore


---

## 🛠 Installation  

### 1️⃣ Clone the Repository  
```bash
git clone git@github.com:thanmaym08/Live-Face-Emotion-Detection.git
cd Live-Face-Emotion-Detection

2️⃣ Create & Activate Virtual Environment

python -m venv venv
source venv/bin/activate   # For Mac/Linux  
venv\Scripts\activate      # For Windows  

3️⃣ Install Dependencies

pip install -r src/requirements.txt

🎯 Usage
1️⃣ Run the Live Emotion Detection

python src/real_time_emotion.py

2️⃣ Train Your Own Model (Optional)

python src/train_model.py

📌 Technologies Used

    Python 3.x
    TensorFlow & Keras (MobileNetV2 for emotion classification)
    OpenCV (Face detection)
    NumPy & Pandas
    Matplotlib & Seaborn (Data visualization)

📷 Demo

🚀 Coming Soon!
📝 License

This project is MIT Licensed.

👤 Created by Thanmay
🔗 GitHub: thanmaym08
