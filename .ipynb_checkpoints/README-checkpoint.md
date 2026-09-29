# 💳 CreditWise Loan Approval System

A **Machine Learning-based Loan Approval Prediction System** that analyzes applicant information and predicts whether a loan application should be approved or rejected.

This project demonstrates a complete machine learning workflow, including **data preprocessing, missing-value handling, exploratory data analysis (EDA), feature encoding, feature scaling, feature engineering, model training, and model evaluation**.

---

## 📌 Project Overview

Loan approval decisions depend on several factors such as income, credit score, debt-to-income ratio, education, employment status, loan amount, and other applicant information.

The goal of this project is to use historical loan application data to build machine learning classification models that can predict the `Loan_Approved` outcome.

### 🎯 Objective

* Analyze loan applicant data
* Handle missing values
* Perform exploratory data analysis
* Convert categorical data into numerical form
* Scale numerical features
* Analyze relationships between features
* Train multiple machine learning classification models
* Evaluate model performance
* Perform feature engineering to improve the models

---

## 🛠️ Technologies Used

| Technology       | Purpose                        |
| ---------------- | ------------------------------ |
| Python           | Programming language           |
| NumPy            | Numerical operations           |
| Pandas           | Data manipulation and analysis |
| Matplotlib       | Data visualization             |
| Seaborn          | Statistical visualization      |
| Scikit-learn     | Machine Learning               |
| Jupyter Notebook | Development environment        |

---

## 🤖 Machine Learning Models

The project uses the following classification algorithms:

### 1. Logistic Regression

Used as a baseline classification model for predicting loan approval.

### 2. K-Nearest Neighbors (KNN)

Classifies applicants based on the characteristics of nearby data points.

### 3. Gaussian Naive Bayes

Uses probability-based classification to predict loan approval.

---

## 📊 Dataset

The project uses:

`loan_approval_data1.csv`

The dataset contains applicant and loan-related information.

Important features include:

* Applicant ID
* Gender
* Applicant Income
* Credit Score
* Education Level
* Employment Status
* Marital Status
* Loan Amount
* Loan Purpose
* Property Area
* DTI Ratio
* Savings
* Employer Category
* Loan Approved

### Target Variable

```text
Loan_Approved
```

The target represents whether the applicant's loan was approved.

---

## 🔄 Machine Learning Workflow

The project follows these major steps:

```text
Dataset
   ↓
Data Loading
   ↓
Data Inspection
   ↓
Missing Value Handling
   ↓
Exploratory Data Analysis
   ↓
Categorical Feature Encoding
   ↓
Correlation Analysis
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
Model Training
   ↓
Model Prediction
   ↓
Model Evaluation
   ↓
Feature Engineering
   ↓
Improved Model Preparation
```

---

## 🧹 1. Data Preprocessing

The dataset is first loaded using Pandas.

```python
df = pd.read_csv("loan_approval_data1.csv")
```

The dataset is inspected using:

```python
df.info()
df.describe()
```

### Missing Values

Numerical missing values are handled using the **mean**:

```python
num_imp = SimpleImputer(strategy="mean")
df[numerical_cols] = num_imp.fit_transform(df[numerical_cols])
```

Categorical missing values are handled using the **most frequent value**:

```python
cat_imp = SimpleImputer(strategy="most_frequent")
df[categorical_cols] = cat_imp.fit_transform(df[categorical_cols])
```

---

## 📈 2. Exploratory Data Analysis

EDA is performed to understand the dataset and identify patterns.

The project includes:

* Loan approval distribution
* Gender distribution
* Education-level distribution
* Applicant income distribution
* Outlier analysis
* Credit score analysis
* DTI ratio analysis
* Savings analysis
* Correlation analysis

### Visualizations

The project uses:

* Pie charts
* Bar charts
* Histograms
* Box plots
* Correlation heatmap

Example:

```python
sns.histplot(
    data=df,
    x="Credit_Score",
    hue="Loan_Approved",
    bins=20,
    multiple="dodge"
)
```

---

## 🔤 3. Feature Encoding

Machine learning models require numerical input, so categorical variables are converted into numerical features.

### Label Encoding

`Education_Level` and `Loan_Approved` are encoded using `LabelEncoder`.

```python
le = LabelEncoder()

df["Education_Level"] = le.fit_transform(df["Education_Level"])
df["Loan_Approved"] = le.fit_transform(df["Loan_Approved"])
```

### One-Hot Encoding

Other categorical features are converted using `OneHotEncoder`.

```python
cols = [
    "Employment_Status",
    "Marital_Status",
    "Loan_Purpose",
    "Property_Area",
    "Gender",
    "Employer_Category"
]
```

---

## 📊 4. Correlation Analysis

A correlation matrix is created to understand relationships between numerical variables.

```python
corr_matrix = nums_col.corr()

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f"
)
```

The project also checks the correlation of individual features with the target:

```python
nums_col.corr()["Loan_Approved"].sort_values(ascending=False)
```

---

## ✂️ 5. Train-Test Split

The dataset is divided into training and testing data.

```python
X = df.drop("Loan_Approved", axis=1)
y = df["Loan_Approved"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

Here:

* **80%** of the data is used for training
* **20%** is used for testing

---

## ⚖️ 6. Feature Scaling

StandardScaler is used to scale the features.

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Scaling is especially useful for algorithms such as **KNN** and Logistic Regression.

---

## 🧠 7. Model Training

### Logistic Regression

```python
log_model = LogisticRegression()

log_model.fit(X_train_scaled, y_train)

y_pred = log_model.predict(X_test_scaled)
```

### KNN

```python
knn_model = KNeighborsClassifier(n_neighbors=5)

knn_model.fit(X_train_scaled, y_train)

y_pred = knn_model.predict(X_test_scaled)
```

### Gaussian Naive Bayes

```python
naive_model = GaussianNB()

naive_model.fit(X_train_scaled, y_train)

y_pred = naive_model.predict(X_test_scaled)
```

---

## 📏 8. Model Evaluation

The models are evaluated using several classification metrics:

### Accuracy

Measures the percentage of correct predictions.

### Precision

Measures how many predicted approvals were actually correct.

### Recall

Measures how many actual positive cases were correctly identified.

### F1 Score

Combines precision and recall into a single metric.

### Confusion Matrix

Shows:

* True Positive
* True Negative
* False Positive
* False Negative

Example:

```python
print("Precision:", precision_score(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Confusion Matrix:", confusion_matrix(y_test, y_pred))
```

---

## 🧪 9. Feature Engineering

Additional features are created to help the model capture non-linear relationships.

### DTI Ratio Squared

```python
df["DTI_Ratio_sq"] = df["DTI_Ratio"] ** 2
```

### Credit Score Squared

```python
df["Credit_Score_sq"] = df["Credit_Score"] ** 2
```

### Log Transformation of Income

```python
df["Applicant_Income_log"] = np.log1p(
    df["Applicant_Income"]
)
```

These transformations can help the model represent relationships that may not be captured well by the original features.

---

## 📁 Project Structure

```text
Lone_Approval_System/
│
├── credit_wise.ipynb
├── loan_approval_data1.csv
├── 30. CreditWise_Loan_System.txt
└── README.md
```

### Files

**`credit_wise.ipynb`**

Main Jupyter Notebook containing:

* Data preprocessing
* EDA
* Visualization
* Feature encoding
* Feature scaling
* Model training
* Model evaluation
* Feature engineering

**`loan_approval_data1.csv`**

Dataset used for training and testing the machine learning models.

**`30. CreditWise_Loan_System.txt`**

Project-related notes/resources.

---

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Lone_Approval_System.git
```

### 2. Open the project folder

```bash
cd Lone_Approval_System
```

### 3. Install required libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 4. Start Jupyter Notebook

```bash
jupyter notebook
```

### 5. Open

```text
credit_wise.ipynb
```

### 6. Run the notebook

Run the cells from top to bottom.

---

## 📌 Key Concepts Demonstrated

This project covers several important Data Science and Machine Learning concepts:

* Data Collection
* Data Cleaning
* Missing Value Imputation
* Exploratory Data Analysis
* Data Visualization
* Categorical Encoding
* One-Hot Encoding
* Label Encoding
* Feature Scaling
* Correlation Analysis
* Train-Test Split
* Classification
* Logistic Regression
* KNN
* Naive Bayes
* Model Evaluation
* Confusion Matrix
* Feature Engineering
* Log Transformation

---

## 🔮 Future Improvements

The project can be extended with:

* Hyperparameter tuning using GridSearchCV
* Cross-validation
* Random Forest
* Decision Tree
* Support Vector Machine
* XGBoost
* ROC-AUC evaluation
* Class imbalance handling
* Feature selection
* Model comparison
* Model serialization using Joblib
* Streamlit web application
* Real-time loan prediction interface
* Deployment as a web application

---

## 👨‍💻 Author

**Fenil Dhanani**

MCA Student | Python | Data Science | Machine Learning | AI/ML

---

## ⭐ Project Goal

The main goal of this project is to understand how a real-world dataset can be transformed into a machine learning solution through:

**Data → Cleaning → Analysis → Visualization → Preprocessing → Machine Learning → Evaluation → Feature Engineering**

---

## 📜 License

This project is created for educational and learning purposes.
