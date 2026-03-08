# Insurance Claims Fraud Detection System

## Epic 1: Problem Definition

### 1.1 Business Problem
Insurance fraud is one of the most significant challenges in the insurance industry, costing billions of dollars annually. Fraudulent claims inflate premiums for honest customers and drain company resources. This project builds a **Machine Learning classification system** to automatically detect potentially fraudulent auto insurance claims based on policy, customer, incident, and claim data.

### 1.2 Business Requirements
- Analyze historical insurance claim data with fraud labels
- Identify key features that distinguish fraudulent from genuine claims
- Train and evaluate multiple ML models for binary classification
- Deploy a web-based prediction tool for insurance investigators

### 1.3 Literature Survey
| Technique | Reference |
|---|---|
| Logistic Regression for fraud scoring | Phua et al., "A Comprehensive Survey of Data Mining-based Fraud Detection Research" |
| Random Forest ensemble methods | Breiman, L. (2001). "Random Forests", Machine Learning |
| Gradient Boosting for imbalanced classification | Friedman, J. (2001). "Greedy Function Approximation" |
| Feature engineering for insurance data | Baesens et al., "Fraud Analytics Using Descriptive, Predictive, and Social Network Techniques" |

### 1.4 Social & Business Impact
- **Cost reduction**: Automated screening reduces manual investigation workload by up to 70%
- **Premium fairness**: Detecting fraud keeps premiums lower for honest policyholders
- **Faster processing**: Genuine claims are processed faster when fraud cases are filtered out
- **Deterrence**: Visible fraud detection systems discourage future fraudulent attempts

---

## Dataset Features

| Category | Features |
|---|---|
| **Policy** | months_as_customer, age, policy_state, policy_csl, policy_deductable, policy_annual_premium, umbrella_limit |
| **Insured Person** | insured_sex, insured_education_level, insured_occupation, insured_hobbies, insured_relationship, capital-gains, capital-loss |
| **Incident** | incident_type, collision_type, incident_severity, authorities_contacted, incident_state, incident_hour_of_the_day, number_of_vehicles_involved, property_damage, bodily_injuries, witnesses, police_report_available |
| **Claim & Vehicle** | total_claim_amount, injury_claim, property_claim, vehicle_claim, auto_make, auto_year |
| **Target** | fraud_reported (Y / N) |

---

## Project Structure

```
insurance-fraud-detection/
├── generate_dataset.py          # Dataset generation script
├── insurance_fraud_dataset.csv  # Dataset (1000 rows × 40 columns)
├── fraud_detection.py           # ML pipeline (Epics 2–5)
├── fraud_detection.ipynb        # Interactive Jupyter Notebook
├── app.py                       # Flask web application (Epic 6)
├── best_model.pkl               # Saved best model
├── scaler.pkl                   # Saved StandardScaler
├── label_encoders.pkl           # Saved LabelEncoders
├── feature_columns.pkl          # Feature column order
├── requirements.txt             # Python dependencies
├── templates/
│   ├── index.html               # Prediction input form
│   └── result.html              # Prediction result page
├── static/
│   └── style.css                # Premium dark-mode CSS
├── plots/
│   ├── 00_requested_boxplots.png
│   ├── 01_target_distribution.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_claim_distributions.png
│   ├── 04_boxplots.png
│   ├── 05_incident_type_fraud.png
│   ├── 06_model_comparison.png
│   ├── 07_confusion_matrices.png
│   ├── 08_incident_severity_pie.png
│   └── 09_age_histogram.png
└── README.md
```

---

## Epic 3: Exploratory Data Analysis

### 3.1 Univariate Analysis
Univariate analysis involves understanding the data by examining each feature individually. In this project, we utilize various visualizations to gain insights into the distribution and composition of key features.

#### Fraud Distribution (Countplot)
Using the `countplot` function from the Seaborn package, we can visualize the number of unique values in categorical features.
- **Insight**: Out of 1000 insurance claims, there are only **247 fraud cases reported** (28.5% in training), indicating an imbalanced dataset which we address during model training.

#### Incident Severity Composition (Pie Chart)
The pie chart describes the composition of the `incident_severity` feature, showing how different levels of damage are distributed.
- **Insight**: 
  - **35.4%** of cases are Minor Damage.
  - Major Damage and Total Loss have almost equal compositions.
  - Only **9%** of cases are Trivial Damage.

### 3.2 Multivariate Analysis
Multivariate analysis is used to find the relation between multiple features. We utilized Seaborn heatmaps to identify highly correlated features.

- **Findings**:
    - **Age & Months as Customer**: Highly correlated (0.92).
    - **Claim Amounts**: `injury_claim`, `property_claim`, and `vehicle_claim` are highly correlated with `total_claim_amount` (0.85-0.98).
- **Action**: These highly correlated features (`age`, `injury_claim`, `property_claim`, `vehicle_claim`) were dropped to reduce multicollinearity and simplify the model.

![Correlation Heatmap](file:///Users/ram/.gemini/antigravity/scratch/plots/02_correlation_heatmap.png)

---

---

## Step-by-Step Procedure

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Provide Dataset
Place your own CSV file as `insurance_fraud_dataset.csv` in the root directory.

---

## Epic 2: Data Collection & Preparation

### 2.1 Data Ingestion
The dataset can be in various formats such as **.csv, Excel, .txt, or .json**. In this project, we utilize the powerful **Pandas** library to read the data.
- **Method**: The `read_csv()` function is used to load the dataset by providing the directory or filename as a parameter.
- **Format**: `df = pd.read_csv("insurance_fraud_dataset.csv")`

### 2.2 Handling Missing Values
For checking the null values, `df.isna().any()` function is used. To sum those null values we use `.sum()` function. From our analysis (referencing results below), we found that there are no null values present in our dataset. So we can skip handling the missing values step.

### 2.3 Handling Outliers
With the help of **Boxplots**, outliers are visualized. In this project, we identify the upper and lower bounds of features like `policy_annual_premium` and `months_as_customer` using mathematical formulas.

- **Visualization**: Boxplots from the Seaborn library are used to visualize the distribution and identify points beyond the whiskers.
- **Interquartile Range (IQR) Method**:
  - **Upper Bound**: Multiply IQR by 1.5 and add it to the 3rd quartile ($Q3 + 1.5 \times IQR$).
  - **Lower Bound**: Multiply IQR by 1.5 and subtract it from the 1st quartile ($Q1 - 1.5 \times IQR$).
- **Transformation Technique**: To handle identified outliers, **Log Transformation** (`np.log1p`) is applied. This technique reduces skewness and stabilizes variance.
- **Verification**: We use a custom function to visualize the distribution and probability plots before and after transformation to ensure the data is better distributed for modeling.

---

---

### Step 3: Run the ML Pipeline
You can run the pipeline either as a Python script or interactively via Jupyter Notebook:

**Option A: Python Script**
```bash
python3 fraud_detection.py
```

**Option B: Jupyter Notebook**
```bash
jupyter notebook fraud_detection.ipynb
```

This script/notebook will:
1. Load and clean the dataset (handle `?` values, drop irrelevant columns)
2. Label-encode categorical features
3. **Drop highly correlated features** (`age`, `injury_claim`, `property_claim`, `vehicle_claim`) identified through multivariate analysis
4. Print descriptive statistics
5. Save 9 EDA plots to `plots/` (including Correlation Heatmap and Incident Severity Pie Chart)
6. Train 4 models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting)
7. Evaluate with accuracy, precision, recall, F1, confusion matrix, cross-validation
8. Perform hyperparameter tuning with GridSearchCV
9. Save the best model as `best_model.pkl`

### Step 4: Launch the Web Application
```bash
python app.py
```
Open `http://127.0.0.1:5000` in your browser. Fill in the form and click **Predict Fraud**.

---

## Models Trained

| # | Algorithm | Description |
|---|---|---|
| 1 | Logistic Regression | Linear baseline; fast and interpretable |
| 2 | Decision Tree | Non-linear; captures feature interactions |
| 3 | Random Forest | Ensemble of trees; robust to overfitting |
| 4 | Gradient Boosting | Sequential boosting; strong on tabular data |

---

## Key Techniques Used

- **Label Encoding** — converts categorical features to numerical values
- **Standard Scaling** — normalizes features to mean=0, std=1
- **Train/Test Split** — 80/20 split with stratification
- **Cross-Validation** — 5-fold CV for reliable performance estimates
- **GridSearchCV** — exhaustive hyperparameter search for Random Forest & Gradient Boosting
- **Confusion Matrix, Precision, Recall, F1** — comprehensive evaluation metrics
