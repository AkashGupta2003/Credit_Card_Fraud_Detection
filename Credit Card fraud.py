# -------------------- Import Libraries --------------------
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# -------------------- Load and Preprocess Dataset --------------------
Dataset = pd.read_csv('creditcard.csv')
pd.options.display.max_columns = None

# Standardize 'Amount' and drop 'Time'
sc = StandardScaler()
Dataset['Amount'] = sc.fit_transform(pd.DataFrame(Dataset['Amount']))
Dataset.drop('Time', axis=1, inplace=True)

# Remove duplicates
Dataset.drop_duplicates(inplace=True)

# -------------------- Under Sampling --------------------
normal = Dataset[Dataset['Class'] == 0]
fraud = Dataset[Dataset['Class'] == 1]
normal_sample = normal.sample(n=len(fraud))
under_data = pd.concat([normal_sample, fraud], ignore_index=True)

X_under = under_data.drop('Class', axis=1).values
y_under = under_data['Class'].values

X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(X_under, y_under, test_size=0.2, random_state=42)

# -------------------- Over Sampling with SMOTE --------------------
X = Dataset.drop('Class', axis=1).values
y = Dataset['Class'].values

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# -------------------- Train Model (RandomForest) on Over Sampled Data --------------------
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_o, y_train_o)
y_pred = rf.predict(X_test_o)
y_prob = rf.predict_proba(X_test_o)[:, 1]

# -------------------- Evaluation Metrics --------------------
print("Accuracy:", accuracy_score(y_test_o, y_pred))
print("Recall:", recall_score(y_test_o, y_pred))
print("Precision:", precision_score(y_test_o, y_pred))
print("F1 Score:", f1_score(y_test_o, y_pred))

# -------------------- Save Model --------------------
joblib.dump(rf, 'Credit_card_fraud_detect.pkl')

# -------------------- ROC Curve --------------------
fpr, tpr, _ = roc_curve(y_test_o, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.2f})'.format(roc_auc), color='darkorange')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.title('ROC Curve - Random Forest')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# -------------------- Precision-Recall Curve --------------------
precision, recall, _ = precision_recall_curve(y_test_o, y_prob)

plt.figure(figsize=(8,6))
plt.plot(recall, precision, label='Precision-Recall Curve', color='blue')
plt.title('Precision-Recall Curve - Random Forest')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid(True)
plt.legend()
plt.show()

# -------------------- Prediction Example --------------------
model = joblib.load('Credit_card_fraud_detect.pkl')
sample_input = np.array([[1]*29])
prediction = model.predict(sample_input)

print("Prediction:", "Fraud" if prediction[0] == 1 else "Normal")
