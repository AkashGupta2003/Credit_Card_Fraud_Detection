<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Credit Card Fraud Detection</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 40px;
            background-color: #f4f4f9;
            color: #333;
            line-height: 1.6;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        .section {
            margin-bottom: 30px;
            padding: 20px;
            background-color: white;
            border-left: 6px solid #3498db;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        ul {
            padding-left: 20px;
        }
        a {
            color: #2980b9;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>

    <h1>🧠 Credit Card Fraud Detection</h1>

    <div class="section">
        <h2>🎯 Objective</h2>
        <p>To analyze credit card transactions and develop machine learning models that accurately detect fraudulent transactions.</p>
    </div>

    <div class="section">
        <h2>📌 Overview</h2>
        <p>
            Credit card fraud detection involves identifying unauthorized or suspicious activities using stolen or compromised card information.
            Financial institutions rely on advanced fraud detection systems to protect customers and minimize financial loss.
        </p>
        <p>
            Dataset Source:
            <a href="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data" target="_blank">
                Kaggle - Credit Card Fraud Detection
            </a>
        </p>
    </div>

    <div class="section">
        <h2>🛠️ Libraries Used</h2>
        <ul>
            <li>NumPy – Numerical computation</li>
            <li>Pandas – Data manipulation and analysis</li>
            <li>Seaborn & Matplotlib – Data visualization</li>
            <li>Scikit-learn – Machine learning and model evaluation</li>
        </ul>
    </div>

    <div class="section">
        <h2>🏗️ Project Workflow</h2>
        <ol>
            <li><strong>Data Collection & Cleaning:</strong> Data sourced from Kaggle. Cleaned by removing irrelevant columns.</li>
            <li><strong>Exploratory Data Analysis (EDA):</strong> Visualized distributions and identified patterns.</li>
            <li><strong>Handling Imbalanced Data:</strong> Used SMOTE, over-sampling, and under-sampling techniques.</li>
            <li><strong>Data Splitting:</strong> Split data into training and testing sets.</li>
            <li><strong>Model Training:</strong> Trained Logistic Regression, Random Forest, and Decision Tree classifiers.</li>
            <li><strong>Model Evaluation:</strong> Metrics used: F1-score, Accuracy, Precision, and Recall.</li>
            <li><strong>Result Visualization:</strong> Plotted confusion matrix, ROC curves, and precision-recall curves.</li>
        </ol>
    </div>

    <div class="section">
        <h2>📈 Business Impact & Use Case</h2>
        <p>
            Traditional rule-based fraud detection systems often result in <strong>high false positive rates</strong>, flagging legitimate transactions.
        </p>
        <p>
            Machine Learning models reduce false positives by <strong>learning individual customer behavior patterns</strong>, resulting in smarter fraud detection with <strong>lower customer friction</strong>.
        </p>
    </div>

</body>
</html>
