
# 📊 Customer Churn Prediction using Machine Learning

---

## 🔍 Project Overview  
This project aims to predict customer churn using a real-world telecom dataset. By identifying customers likely to cancel their subscription, businesses can proactively improve retention and reduce revenue loss.

---

## ✅ Problem Statement  
Customer churn significantly affects subscription-based businesses. The objective is to use customer demographic and usage data to train a machine learning model that accurately predicts churn behavior.

---

## 💾 Dataset  
- **Source:** [Telco Customer Churn Dataset – Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
- **Size:** 7,043 rows × 21 columns  
- **Target Variable:** `Churn` (Yes/No)  
- **Features:** Customer demographics, services subscribed, billing, and tenure details

---

## 🧠 Feature Engineering  
To improve prediction accuracy, the following features were engineered:

- **AvgChargesPerMonth:** TotalCharges / Tenure  
- **num_services_subscribed:** Number of value-added services (e.g., OnlineSecurity, DeviceProtection)  
- **hasStreaming:** Binary flag for streaming services (TV or movies)

---

## ⚖️ Handling Imbalanced Data  
The original dataset had a class imbalance with fewer churners. We applied **SMOTE (Synthetic Minority Over-sampling Technique)** to generate synthetic samples for the minority class, resulting in a significant improvement in recall.

---

## 🛠️ Tools & Technologies  

- **Environment:** Python (Google Colab)  
- **Libraries:**  
  - `pandas`, `numpy` – Data wrangling  
  - `matplotlib`, `seaborn` – Visualization  
  - `scikit-learn` – Modeling, preprocessing, evaluation  
  - `imblearn` – SMOTE  
  - `joblib` – Saving/loading models  
  - `GridSearchCV` – Hyperparameter tuning

---

## 🧪 Machine Learning Models Used  

| Model | Description |
|-------|-------------|
| **Logistic Regression** | Used as a baseline model, then tuned and enhanced with SMOTE |
| **Random Forest Classifier** | Ensemble model capturing non-linear relationships |

---

## 📈 Evaluation Metrics  

- **Accuracy**  
- **Recall** (especially important for churn detection)  
- **Precision**  
- **F1-Score**  
- **Confusion Matrix**  
- **Cross-Validation Scores**

---

## 📊 Logistic Regression Comparison  

| Model Version           | Accuracy | Recall (Churn) | Precision (Churn) | F1-Score (Churn) |
|------------------------|----------|----------------|-------------------|------------------|
| Base Logistic Regression | 0.79     | 0.52           | 0.62              | 0.56             |
| Tuned (GridSearchCV)     | 0.79     | 0.52           | 0.63              | 0.57             |
| **SMOTE-Based**             | 0.74     | **0.80**           | 0.50              | **0.62**             |

---

## ✅ Final Model Selected  
**SMOTE-Based Logistic Regression**  
Chosen for its **high recall (80%)**, ensuring that more churners are identified even at the cost of lower precision — a tradeoff often acceptable in churn prediction scenarios.

---

## 📁 Folder Structure  

```
Customer-Churn-Prediction/
│
├── Customer_Churn_Prediction.ipynb      # Main notebook
├── README.md                   
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset
│
├── models/                     # Saved model files
│   ├── best_logistic_model.pkl
│   └── scaler.pkl
│
└── images/                     # Optional plots and charts
```

---

## 📁 Models Folder  

- **best_logistic_model.pkl** – Final logistic regression model (SMOTE-enhanced)  
- **scaler.pkl** – StandardScaler used on numeric data before model training  

You can load them like this:

```python
import joblib

# Load trained model and scaler
model = joblib.load('models/best_logistic_model.pkl')
scaler = joblib.load('models/scaler.pkl')
```

---

## 🚀 How to Run  

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
2. Upload it to Google Colab or run locally  
3. Open `churn_prediction.ipynb`  
4. Execute each cell step by step to:
   - Preprocess the data  
   - Engineer features  
   - Train & evaluate models  
   - Save the best-performing model  

---

## 📌 Future Work  

- 🚀 **Model Deployment** using **Streamlit** or **Flask**  
- 📊 Create a **dashboard** to monitor churn predictions in real time  
- 🔍 Use **SHAP** for explainable AI and feature impact interpretation  
- 🤖 Try **stacked ensembles** for even better prediction accuracy  

---

## 👩‍💻 Author  

**Nayomi Thilakarathna**  
A fresher in Data Analytics passionate about machine learning, insights generation, and solving real-world business problems.  

---

🌟 If you found this project useful, please consider giving it a ⭐ on GitHub!
