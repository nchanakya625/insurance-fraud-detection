"""
=============================================================================
  INSURANCE CLAIMS FRAUD DETECTION — End-to-End Machine Learning Pipeline
=============================================================================
Covers:
  Epic 2  – Data Collection & Preparation
  Epic 3  – Exploratory Data Analysis (Descriptive + Visual)
  Epic 4  – Model Building (Logistic Regression, Decision Tree, Random Forest, XGBoost)
  Epic 5  – Performance Testing & Hyperparameter Tuning
  Model and Scaler are saved for Flask deployment (Epic 6).
"""

import os, warnings, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import scipy.stats as stats

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  EPIC 2 — DATA COLLECTION & PREPARATION
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  EPIC 2: DATA COLLECTION & PREPARATION")
print("=" * 70)

df = pd.read_csv("insurance_fraud_dataset.csv")
print(f"\n✅ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns\n")

# --- 2.0  Visualizing raw features (User Requested Boxplots) ---
print("Generating requested boxplots for policy_number and months_as_customer...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

if "policy_number" in df.columns:
    sns.boxplot(y=df["policy_number"], ax=axes[0], color="#9b59b6")
    axes[0].set_title("Boxplot: Policy Number", fontweight="bold")

if "months_as_customer" in df.columns:
    sns.boxplot(y=df["months_as_customer"], ax=axes[1], color="#3498db")
    axes[1].set_title("Boxplot: Months as Customer", fontweight="bold")

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/00_requested_boxplots.png", dpi=150)
plt.close()
print("  📊 Saved 00_requested_boxplots.png\n")

# --- 2.1  Drop irrelevant and highly correlated columns ---
drop_cols = ["_c39", "policy_number", "policy_bind_date", "incident_date",
             "incident_location", "insured_zip", "incident_city", "auto_model"]
# Highly correlated features identified during multivariate analysis:
# - age: highly correlated with months_as_customer (0.92)
# - injury_claim, property_claim, vehicle_claim: highly correlated with total_claim_amount
corr_drop_cols = ["age", "injury_claim", "property_claim", "vehicle_claim"]
final_drop = drop_cols + corr_drop_cols

df.drop(columns=[c for c in final_drop if c in df.columns], inplace=True)
print(f"Dropped columns: {final_drop}")
print(f"Remaining: {df.shape[1]} columns\n")

# --- 2.2  Handle missing / '?' values ---
# For checking the null values, df.isna().any() function is used.
# To sum those null values we use .sum() function.
# From the analysis, we found that there are no null values present in our dataset.
# So we can skip handling the missing values step.

missing_values = df.isna().sum()
print("Null values per column (isna().sum()):\n", missing_values[missing_values > 0])

# As per observation, if there are no literal null values, we still handle '?' if present.
for col in df.columns:
    df[col].replace("?", np.nan, inplace=True)

# Re-check after replacing '?'
print("\nMissing values after handling '?' (isna().sum()):")
print(df.isna().sum().sum(), "total missing values found.")

# The dataset is clean (no nulls), but we keep the logic for robustness.
# Fill categorical NaN with mode
for col in df.select_dtypes(include="object").columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

# Fill numeric NaN with median
for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

print("\n✅ Data check complete (No null values found or handled)\n")

# --- 2.3  Encode target ---
df["fraud_reported"] = df["fraud_reported"].map({"Y": 1, "N": 0})

# --- 2.4  Label-encode categorical features ---
label_encoders = {}
cat_cols = df.select_dtypes(include="object").columns.tolist()
print(f"Encoding {len(cat_cols)} categorical columns: {cat_cols}\n")

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

print(df.head())

# --- 2.5  Handling Outliers ---
print("\n--- Handling Outliers ---")

def plot_distribution_prob(df, feature, title_prefix=""):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram + KDE
    sns.histplot(df[feature], kde=True, ax=axes[0], color="skyblue")
    axes[0].set_title(f"{title_prefix} Distribution of {feature}", fontweight="bold")
    
    # Probability Plot
    stats.probplot(df[feature], dist="norm", plot=axes[1])
    axes[1].set_title(f"{title_prefix} Probability Plot", fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{feature}_{title_prefix.lower().replace(' ', '_')}_dist.png", dpi=150)
    plt.close()
    print(f"  📊 Saved {feature}_{title_prefix.lower().replace(' ', '_')}_dist.png")

# Visualizing outliers with boxplot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(y=df["policy_annual_premium"], ax=axes[0], color="#D4A5FF")
axes[0].set_title("Boxplot: Policy Annual Premium", fontweight="bold")
sns.boxplot(y=df["months_as_customer"], ax=axes[1], color="#A5D6FF")
axes[1].set_title("Boxplot: Months as Customer", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/10_outliers_boxplot_before.png", dpi=150)
plt.close()

# Distribution before transformation
plot_distribution_prob(df, "policy_annual_premium", "Before")

# Calculate bounds using IQR method
for col in ["policy_annual_premium", "months_as_customer"]:
    if col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        print(f"Feature: {col}")
        print(f"  IQR: {IQR:.2f} | Lower Bound: {lower_bound:.2f} | Upper Bound: {upper_bound:.2f}")

# Handle outliers using Log Transformation
# Note: Log transformation helps normalize skewed data and reduces the impact of outliers.
df["policy_annual_premium"] = np.log1p(df["policy_annual_premium"])
df["months_as_customer"]    = np.log1p(df["months_as_customer"])

print("✅ Log transformation applied to policy_annual_premium & months_as_customer")

# Distribution after transformation
plot_distribution_prob(df, "policy_annual_premium", "After")

# Visualizing after transformation
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(y=df["policy_annual_premium"], ax=axes[0], color="#D4A5FF")
axes[0].set_title("Boxplot (After Log): Policy Annual Premium", fontweight="bold")
sns.boxplot(y=df["months_as_customer"], ax=axes[1], color="#A5D6FF")
axes[1].set_title("Boxplot (After Log): Months as Customer", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/11_outliers_boxplot_after.png", dpi=150)
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
#  EPIC 2 (Continued) — Final Check
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\nFinal dataset: {df.shape[0]} rows × {df.shape[1]} columns")

# ═══════════════════════════════════════════════════════════════════════════════
#  EPIC 3 — EXPLORATORY DATA ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  EPIC 3: EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# --- 3.1  Descriptive Statistics ---
print("\n── Descriptive Statistics ──")
print(df.describe().round(2).to_string())
print(f"\nTarget distribution:\n{df['fraud_reported'].value_counts().to_string()}")

# --- 3.2  Visual Analysis ---

# Plot 1: Target distribution
fig, ax = plt.subplots(figsize=(6, 4))
counts = df["fraud_reported"].value_counts()
bars = ax.bar(["Not Fraud (0)", "Fraud (1)"], counts.values,
              color=["#2ecc71", "#e74c3c"], edgecolor="black")
for b in bars:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 10,
            str(int(b.get_height())), ha="center", fontweight="bold")
ax.set_title("Fraud vs Non-Fraud Distribution", fontsize=14, fontweight="bold")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/01_target_distribution.png", dpi=150)
plt.close()
print("  📊 Saved 01_target_distribution.png")

# Plot 2: Correlation heatmap
fig, ax = plt.subplots(figsize=(14, 10))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=False, cmap="coolwarm", center=0,
            linewidths=0.5, ax=ax)
ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/02_correlation_heatmap.png", dpi=150)
plt.close()
print("  📊 Saved 02_correlation_heatmap.png")

# Plot 3: Claim amount distributions
fig, ax = plt.subplots(figsize=(8, 5))
if "total_claim_amount" in df.columns:
    sns.histplot(data=df, x="total_claim_amount", hue="fraud_reported", kde=True, ax=ax,
                 palette={0: "#2ecc71", 1: "#e74c3c"}, alpha=0.6)
    ax.set_title("Total Claim Amount Distribution by Fraud Status", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/03_claim_distributions.png", dpi=150)
plt.close()
print("  📊 Saved 03_claim_distributions.png")

# Plot 4: Box plots for key numeric features
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
box_cols = [c for c in ["months_as_customer", "policy_deductable", "policy_annual_premium",
                          "capital-gains", "incident_hour_of_the_day",
                          "total_claim_amount"] if c in df.columns]
for ax, col in zip(axes.flat, box_cols):
    sns.boxplot(data=df, x="fraud_reported", y=col, ax=ax,
                palette=["#3498db", "#e74c3c"])
    ax.set_title(col.replace("_", " ").title(), fontweight="bold")
    ax.set_xticklabels(["Not Fraud", "Fraud"])
plt.suptitle("Feature Box Plots: Fraud vs Non-Fraud", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/04_boxplots.png", dpi=150)
plt.close()
print("  📊 Saved 04_boxplots.png")

# Plot 5: Incident type vs fraud
if "incident_type" in df.columns:
    fig, ax = plt.subplots(figsize=(8, 4))
    ct = pd.crosstab(df["incident_type"], df["fraud_reported"])
    ct.plot(kind="bar", stacked=True, color=["#2ecc71", "#e74c3c"],
            edgecolor="black", ax=ax)
    ax.set_title("Incident Type vs Fraud", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count")
    ax.legend(["Not Fraud", "Fraud"])
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/05_incident_type_fraud.png", dpi=150)
    plt.close()
    print("  📊 Saved 05_incident_type_fraud.png")

# Plot 8: Incident severity composition (Pie Chart)
if "incident_severity" in df.columns:
    # We need the original labels for the pie chart, but df is already encoded.
    # We can use the label_encoder to decode or just use the encoded values if we knew the map.
    # Since we want a "premium" look, let's use the encoder we saved earlier if available, 
    # or just use value_counts from the encoded column for now.
    fig, ax = plt.subplots(figsize=(8, 8))
    # Note: df is label encoded, but for the pie chart we'd prefer labels.
    # Let's use the counts and labels from the encoded column.
    counts = df["incident_severity"].value_counts()
    le = label_encoders["incident_severity"]
    labels = le.inverse_transform(counts.index)
    
    colors = sns.color_palette("pastel")[0:len(counts)]
    ax.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140, 
           colors=colors, wedgeprops={'edgecolor': 'black'})
    ax.set_title("Incident Severity Composition", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/08_incident_severity_pie.png", dpi=150)
    plt.close()
    print("  📊 Saved 08_incident_severity_pie.png")

# Plot 9: Total Claim Amount Distribution (Histogram)
if "total_claim_amount" in df.columns:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["total_claim_amount"], kde=True, color="#3498db", ax=ax, bins=15)
    ax.set_title("Total Claim Amount Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Total Claim Amount")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/09_claim_amount_histogram.png", dpi=150)
    plt.close()
    print("  📊 Saved 09_claim_amount_histogram.png")

# ═══════════════════════════════════════════════════════════════════════════════
#  EPIC 4 — MODEL BUILDING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  EPIC 4: MODEL BUILDING")
print("=" * 70)

# --- 4.1  Train / Test split ---
X = df.drop("fraud_reported", axis=1)
y = df["fraud_reported"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test  set: {X_test.shape[0]} samples\n")

# --- 4.2  Feature scaling ---
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# --- 4.3  Train multiple models ---
models = {
    "Logistic Regression":  LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":        DecisionTreeClassifier(random_state=42),
    "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":    GradientBoostingClassifier(n_estimators=100, random_state=42),
}

results = {}
print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-" * 70)

for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    results[name] = {"accuracy": acc, "precision": prec, "recall": rec,
                      "f1": f1, "y_pred": y_pred, "model": model}
    print(f"{name:<25} {acc:>10.4f} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
#  EPIC 5 — PERFORMANCE TESTING & HYPERPARAMETER TUNING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  EPIC 5: PERFORMANCE TESTING & HYPERPARAMETER TUNING")
print("=" * 70)

# --- 5.1  Confusion matrices ---
print("\n── Confusion Matrices ──")
for name, r in results.items():
    cm = confusion_matrix(y_test, r["y_pred"])
    print(f"\n{name}:")
    print(cm)

# --- 5.2  Cross-validation ---
print("\n── Cross-Validation Scores (5-fold) ──")
print(f"{'Model':<25} {'Mean CV Score':>15} {'Std':>10}")
print("-" * 55)
for name, r in results.items():
    cv = cross_val_score(r["model"], X_train_sc, y_train, cv=5, scoring="accuracy")
    print(f"{name:<25} {cv.mean():>15.4f} {cv.std():>10.4f}")

# --- 5.3  Classification reports ---
print("\n── Detailed Classification Reports ──")
for name, r in results.items():
    print(f"\n{'─' * 40}")
    print(f"  {name}")
    print(f"{'─' * 40}")
    print(classification_report(y_test, r["y_pred"], target_names=["Not Fraud", "Fraud"]))

# --- 5.4  Hyperparameter Tuning (Random Forest) ---
print("\n── Hyperparameter Tuning: Random Forest (GridSearchCV) ──")

rf_before_acc = results["Random Forest"]["accuracy"]
rf_before_f1  = results["Random Forest"]["f1"]
print(f"\nBEFORE tuning — Accuracy: {rf_before_acc:.4f}  |  F1: {rf_before_f1:.4f}")

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=0,
)
grid_search.fit(X_train_sc, y_train)
best_rf = grid_search.best_estimator_
print(f"Best params: {grid_search.best_params_}")

y_pred_tuned = best_rf.predict(X_test_sc)
rf_after_acc = accuracy_score(y_test, y_pred_tuned)
rf_after_f1  = f1_score(y_test, y_pred_tuned)
print(f"AFTER  tuning — Accuracy: {rf_after_acc:.4f}  |  F1: {rf_after_f1:.4f}")

print(f"\n  Accuracy change: {rf_after_acc - rf_before_acc:+.4f}")
print(f"  F1 change:       {rf_after_f1 - rf_before_f1:+.4f}")

# --- 5.5  Hyperparameter Tuning (Gradient Boosting) ---
print("\n── Hyperparameter Tuning: Gradient Boosting (GridSearchCV) ──")
gb_before_acc = results["Gradient Boosting"]["accuracy"]
gb_before_f1  = results["Gradient Boosting"]["f1"]
print(f"\nBEFORE tuning — Accuracy: {gb_before_acc:.4f}  |  F1: {gb_before_f1:.4f}")

gb_param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.8, 1.0],
}

gb_grid = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    gb_param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=0,
)
gb_grid.fit(X_train_sc, y_train)
best_gb = gb_grid.best_estimator_
print(f"Best params: {gb_grid.best_params_}")

y_pred_gb_tuned = best_gb.predict(X_test_sc)
gb_after_acc = accuracy_score(y_test, y_pred_gb_tuned)
gb_after_f1  = f1_score(y_test, y_pred_gb_tuned)
print(f"AFTER  tuning — Accuracy: {gb_after_acc:.4f}  |  F1: {gb_after_f1:.4f}")

print(f"\n  Accuracy change: {gb_after_acc - gb_before_acc:+.4f}")
print(f"  F1 change:       {gb_after_f1 - gb_before_f1:+.4f}")

# --- 5.6  Model comparison plot ---
model_names = list(results.keys()) + ["RF (Tuned)", "GB (Tuned)"]
accuracies  = [r["accuracy"] for r in results.values()] + [rf_after_acc, gb_after_acc]
f1_scores   = [r["f1"] for r in results.values()] + [rf_after_f1, gb_after_f1]

fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(model_names))
w = 0.35
bars1 = ax.bar(x - w/2, accuracies, w, label="Accuracy", color="#3498db", edgecolor="black")
bars2 = ax.bar(x + w/2, f1_scores,  w, label="F1-Score", color="#e74c3c", edgecolor="black")
for b in list(bars1) + list(bars2):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
            f"{b.get_height():.3f}", ha="center", fontsize=8, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=25, ha="right")
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score")
ax.set_title("Model Comparison: Accuracy & F1-Score", fontsize=14, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/06_model_comparison.png", dpi=150)
plt.close()
print(f"\n  📊 Saved 06_model_comparison.png")

# --- 5.7  Confusion matrix plots ---
fig, axes = plt.subplots(1, 4, figsize=(20, 4))
for ax, (name, r) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, r["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Not Fraud", "Fraud"],
                yticklabels=["Not Fraud", "Fraud"])
    ax.set_title(name, fontweight="bold")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
plt.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/07_confusion_matrices.png", dpi=150)
plt.close()
print("  📊 Saved 07_confusion_matrices.png")

# ═══════════════════════════════════════════════════════════════════════════════
#  SAVE BEST MODEL
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  SAVING BEST MODEL")
print("=" * 70)

# Pick whichever tuned model has the best F1
if gb_after_f1 >= rf_after_f1:
    best_model = best_gb
    best_name  = "Gradient Boosting (Tuned)"
    best_f1    = gb_after_f1
else:
    best_model = best_rf
    best_name  = "Random Forest (Tuned)"
    best_f1    = rf_after_f1

print(f"\n🏆 Best model: {best_name}  (F1 = {best_f1:.4f})")

joblib.dump(best_model, "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(list(X.columns), "feature_columns.pkl")

print("  ✅ Saved best_model.pkl")
print("  ✅ Saved scaler.pkl")
print("  ✅ Saved label_encoders.pkl")
print("  ✅ Saved feature_columns.pkl")
print("\n✅ Pipeline complete! Ready for Flask deployment.\n")
