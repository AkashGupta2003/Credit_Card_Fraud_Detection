# 🧠 Credit Card Fraud Detection

## 🎯 Objective
To analyze credit card transactions and develop machine learning models that accurately detect fraudulent transactions.

---

## 📌 Overview
Credit card fraud detection involves identifying unauthorized or suspicious activities using stolen or compromised card information.  
Financial institutions rely on advanced fraud detection systems to protect customers and minimize financial loss.

**Dataset Source:**  
[Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data)

---

## 🛠️ Libraries Used
- **NumPy** – Numerical computation  
- **Pandas** – Data manipulation and analysis  
- **Seaborn** – Data visualization  
- **Matplotlib** – Data visualization  
- **Scikit-learn** – Machine learning and model evaluation  

---

## 🏗️ Project Workflow

1. **Data Collection & Cleaning**  
   - Data sourced from Kaggle  
   - Cleaned by removing irrelevant or redundant columns  

2. **Exploratory Data Analysis (EDA)**  
   - Visualized transaction distributions and feature correlations  
   - Identified trends and patterns between fraudulent and normal transactions  

3. **Handling Imbalanced Data**  
   - Dataset is highly imbalanced (fraudulent vs. non-fraudulent)  
   - Applied **SMOTE (Synthetic Minority Oversampling Technique)**  
   - Tried **under-sampling** and **over-sampling** approaches  

4. **Data Splitting**  
   - Split into training and test sets (e.g., 80-20 split)  

5. **Model Training**  
   - Logistic Regression  
   - Random Forest Classifier  
   - Decision Tree Classifier  

6. **Model Evaluation**  
   - Metrics: **F1-score**, **Accuracy**, **Precision**, **Recall**  
   - Compared model performances to select the best performer  

7. **Result Visualization**  
   - Confusion matrix  
   - ROC Curve  
   - Precision-Recall curve  

---

## 📈 Business Impact & Use Case

- Traditional rule-based fraud detection often results in **high false positives**, blocking legitimate transactions (e.g., customer traveling abroad).
- ML models learn **individual behavior patterns**, improving fraud detection accuracy and **reducing false positives**.

