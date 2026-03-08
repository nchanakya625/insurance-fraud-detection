# Project Documentation: Insurance Fraud Detection Using Machine Learning

## Step-by-Step Project Development Procedure

---

### Epic 1: Define Problem / Problem Understanding

**Activity 1.1: Specify the business problem**
Insurance are claimed in order to get a relief amount for any damage cause. Insurance is a means of protection from financial loss, but now-a-days many people are claiming the Insurance by fraud claims. It can be called as a scam. It is called as a fraud claim when a claimant attempts to obtain some benefit or advantage they are not entitled to, or when an insurer knowingly denies some benefit that is due. These types of Insurance claims cause loss to the company. So, it is necessary to detect the claims which are fraud. The number of cases of insurance fraud that are detected is much lower than the number of acts that are actually committed. So the main purpose of the Insurance Fraud Detection system is to predict if the Insurance claim is a Fraud or Legal Claim based on the different appeals and parameters.

**Activity 1.2: Business requirements**
* Analyze historical insurance claim data with fraud labels to identify patterns.
* Identify key features that distinguish fraudulent from genuine claims.
* Train and evaluate multiple Machine Learning models for binary classification.
* User-friendly interface: The classification system should be easy to use and understand for users and investigators.

**Activity 1.3: Literature Survey**
A literature survey for an insurance fraud detection project involves researching and reviewing existing studies, articles, and other publications on the topic of fraud classification. The survey aims to gather information on current classification systems, their strengths and weaknesses, and any gaps in knowledge that the project could address. The literature survey also looks at the methods and techniques used in previous fraud classification projects, and any relevant data or findings that inform the design and implementation of the current project (e.g., handling class imbalances and feature engineering).

**Activity 1.4: Social or Business Impact**
* **Social Impact (Fairness under improved care):** By providing accurate and fast detection of fraud, the project ensures honest policyholders do not suffer increased premiums. Genuine claims are processed much faster, improving the customer experience.
* **Business Model / Impact:** By predicting fraudulent claims, the project assists the insurance company in preventing massive financial losses. It reduces the manual investigation workload and supports effective model-driven development for future policies.

---

### Epic 2: Data Collection & Preparation
**Activity: Data Ingestion and Cleaning**
* The dataset `insurance_fraud_dataset.csv` was collected, containing 1000 records of historical claims across 40 different features.
* The data was loaded using the Pandas library (`pd.read_csv()`).
* Checked for missing values and anomalies (e.g., dealing with '?' characters) to ensure reliable analytics.
* Outliers (e.g., in continuous variables like premium and claim amount) were treated using IQR bounds and Log Transformations.

---

### Epic 3: Exploratory Data Analysis
**Activity: Statistical Visualization**
* **Univariate Analysis:** Generated countplots to see the explicit distribution of fraudulent vs. non-fraudulent flags, revealing a significant imbalance (only ~25% fraud). Plotted pie charts to see the composition of incident severity.
* **Multivariate Analysis:** Plotted a correlation heatmap. Identified highly correlated continuous features like `injury_claim`, `property_claim`, `vehicle_claim`, and `total_claim_amount`. Redundant features were dropped to reduce multicollinearity.

---

### Epic 4 & 5: Model Building and Evaluation
**Activity: Machine Learning Pipeline**
* Categorical data was converted into numerical data using Label Encoding.
* Real-valued data was normalized using `StandardScaler`.
* The dataset was split into Training (80%) and Testing (20%) subsets.
* Evaluated four main algorithms:
  1. Logistic Regression
  2. Decision Tree Classifier
  3. Random Forest Classifier
  4. Gradient Boosting Classifier
* Models were evaluated on Accuracy, Precision, Recall, and F1-Score. Specifically, Grid Search CV was used to optimize hyper-parameters and minimize false negatives in fraud detection.

---

### Epic 6: Flask Web Application
**Activity: Deployment logic**
* The best-performing classification model and pre-processing encoders were saved as `.pkl` files using `joblib`.
* Developed an interactive web application leveraging the Flask framework in Python.
* Designed a responsive UI for users to easily insert claim parameters (Age, Policy details, Claims, Incident details) and receive a real-time prediction (Fraud vs. Genuine).

---

### Epic 7: Project Demonstration & Documentation

**Activity: Project Documentation**
This comprehensive document acts as the complete step-by-step project development procedure, fulfilling the deliverables required for the final project submission.
