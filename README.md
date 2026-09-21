# 🔍 User Behavior Anomaly Detection using Isolation Forest

An unsupervised machine learning project for detecting **abnormal user behavior** using the **Isolation Forest** algorithm.

The project analyzes user telemetry data such as login frequency, failed password attempts, session duration, data downloads, off-hours activity, and unique IP usage to identify potentially anomalous behavior.

---

## 📌 Project Overview

In real-world systems, unusual user activity can indicate:

- 🔐 Account compromise
- 🚨 Suspicious login behavior
- 📥 Abnormally high data downloads
- 🌙 Excessive off-hours activity
- 🌐 Unusual IP usage
- ⚠️ Potential security incidents

Since anomaly detection often works with data where attack labels are unavailable, this project uses **Isolation Forest**, an unsupervised anomaly detection algorithm.

### Project Workflow

```text
User Telemetry Dataset
        ↓
Data Inspection
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Rule-Based Reference Labels
        ↓
Feature Selection
        ↓
Feature Scaling
        ↓
Isolation Forest
        ↓
Base Model Evaluation
        ↓
Parameter Tuning
        ↓
Final Isolation Forest
        ↓
Anomaly Detection
        ↓
Anomaly Score Analysis
        ↓
Detected Anomalies CSV

```

---

# 🎯 Objectives

The main objectives of this project are:

1. Analyze user behavior telemetry.
2. Identify unusual behavior patterns.
3. Apply Isolation Forest for unsupervised anomaly detection.
4. Generate anomaly scores for user activities.
5. Tune important Isolation Forest parameters.
6. Compare model predictions against rule-based reference labels.
7. Evaluate the final model using classification metrics.
8. Export detected anomalies for further investigation.

---

# 📊 Dataset

The project uses a user behavior telemetry dataset containing **600 user records** and **7 columns**.

### Dataset Features

| Feature | Description |
| --- | --- |
| `UserID`                   | Unique identifier for each user                           |
| `Daily_Logins`             | Number of logins performed by the user                    |
| `Failed_Password_Attempts` | Number of failed password attempts                        |
| `Session_Duration_Min`     | User session duration in minutes                          |
| `Data_Download_MB`         | Amount of data downloaded in MB                           |
| `Off_Hours_Activity_Pct`   | Percentage of activity performed during unusual/off-hours |
| `Unique_IP_Count`          | Number of unique IP addresses used                        |

### Dataset Size

```text
Rows    : 600
Columns : 7

```

---

# 🧠 Algorithm Used

## Isolation Forest

Isolation Forest is an **unsupervised anomaly detection algorithm** designed to identify unusual observations.

The basic idea is:

> Anomalies are easier to isolate than normal observations.

Normal users generally have similar behavioral patterns, while unusual users tend to have extreme or uncommon values.

### Example

Consider:

```text
Daily Logins

3
4
5
4
3
4
336  ← suspicious

```

The value `336` is very different from the normal pattern.

Isolation Forest attempts to isolate such unusual observations using random decision trees.

---

# 🔑 Important Isolation Forest Concepts

### `n_estimators`

Number of trees used by the Isolation Forest.

```python
n_estimators=200

```

More trees generally provide a more stable estimation of anomaly scores, at the cost of additional computation.

---

### `max_samples`

Controls the number of samples used to build each tree.

```python
max_samples="auto"

```

---

### `contamination`

Expected proportion of anomalies in the dataset.

Example:

```python
contamination=0.05

```

This represents an expected anomaly proportion of approximately 5%.

---

### `max_features`

Controls the number/proportion of features used when constructing trees.

```python
max_features=1.0

```

---

### `bootstrap`

Controls whether samples are drawn with replacement.

```python
bootstrap=False

```

---

# 🛠️ Technologies Used

### Programming Language

- Python

### Libraries

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

### Machine Learning

- Isolation Forest
- StandardScaler

### Evaluation

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/mdaasik007/user-behavior-anomaly-detection.git

```

Navigate into the project:

```bash
cd user-behavior-anomaly-detection

```

Install the required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn

```

---

# 🚀 How to Run

### 1. Open the Notebook

Open:

```text
User_Behavior_Anomaly_Detection_using_Isolation_Forest.ipynb

```

using:

- Jupyter Notebook
- JupyterLab
- Google Colab
- VS Code

### 2. Upload the Dataset

Place:

```text
user_behavior_telemetry.csv

```

in the appropriate working directory.

### 3. Run the Notebook

Execute the cells sequentially.

The notebook performs:

```text
Data Loading
     ↓
Data Cleaning
     ↓
EDA
     ↓
Reference Label Creation
     ↓
Feature Scaling
     ↓
Isolation Forest
     ↓
Parameter Tuning
     ↓
Final Model
     ↓
Evaluation
     ↓
Anomaly Export

```

---

# 🧹 Data Preprocessing

## 1. Missing Value Check

The dataset is checked for missing values:

```python
df.isnull().sum()

```

---

## 2. Infinite Value Handling

Infinite values are converted into `NaN`:

```python
df.replace(
    [-np.inf, np.inf],
    np.nan,
    inplace=True
)

```

Then missing rows are removed:

```python
df.dropna(inplace=True)

```

---

# 📈 Exploratory Data Analysis

The project performs EDA using:

### Correlation Heatmap

The correlation between numerical behavioral features is visualized.

### Feature Distributions

The following features are analyzed:

```text
Daily_Logins
Failed_Password_Attempts
Session_Duration_Min
Data_Download_MB
Off_Hours_Activity_Pct
Unique_IP_Count

```

Distribution plots help identify unusual values and understand the behavior of the dataset.

---

# 🏷️ Reference Label Generation

Because the dataset does not contain a predefined attack/anomaly label, the project creates a **rule-based reference label**.

The project uses the **3-sigma rule**:

```text
Lower Bound = Mean - 3 × Standard Deviation

Upper Bound = Mean + 3 × Standard Deviation

```

If a user's value falls outside this range for any selected feature:

```text
Reference Label = 1 → Anomaly/Attack

```

Otherwise:

```text
Reference Label = 0 → Normal

```

### Important

These labels are **not ground-truth attack labels**.

They are only used as a reference for evaluating the Isolation Forest predictions.

---

# ⚙️ Feature Selection

The following six behavioral features are used:

```python
features_df = [
    "Daily_Logins",
    "Failed_Password_Attempts",
    "Session_Duration_Min",
    "Data_Download_MB",
    "Off_Hours_Activity_Pct",
    "Unique_IP_Count"
]

```

`UserID` is not used as a machine learning feature because it is an identifier rather than a meaningful behavioral measurement.

---

# 📏 Feature Scaling

The selected features are standardized using `StandardScaler`.

```python
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

```

Scaling transforms the features into a comparable numerical range.

---

# 🌲 Base Isolation Forest Model

The initial model uses:

```python
IsolationForest(
    n_estimators=200,
    max_samples="auto",
    contamination=0.05,
    random_state=42,
    bootstrap=False,
    n_jobs=-1
)

```

The model generates predictions:

```text
 1  → Normal
-1  → Anomaly

```

The predictions are converted into binary labels:

```text
0 → Normal
1 → Anomaly

```

---

# 📊 Anomaly Score

The project also calculates the Isolation Forest anomaly score:

```python
model.decision_function(X_scaled)

```

The score helps understand how strongly a data point is considered normal or anomalous.

The project visualizes the distribution of anomaly scores using a histogram.

---

# 🎛️ Parameter Tuning

Multiple Isolation Forest configurations are tested.

Parameters explored include:

```text
n_estimators
max_samples
contamination
max_features
bootstrap

```

Example configurations include:

```python
n_estimators = 100
n_estimators = 200
n_estimators = 300

```

and different contamination levels such as:

```python
0.05
0.10

```

Each configuration is evaluated using:

```text
Accuracy
Precision
Recall
F1 Score

```

---

# 🏆 Final Model

After testing the parameter combinations, the configuration with the highest **F1 Score against the rule-based reference labels** is selected.

The final Isolation Forest model is then trained using the selected parameters.

```python
final_model = IsolationForest(
    n_estimators=best_params["n_estimators"],
    max_samples=best_params["max_samples"],
    contamination=best_params["contamination"],
    max_features=best_params["max_features"],
    bootstrap=best_params["bootstrap"],
    n_jobs=best_params["n_jobs"],
    random_state=42
)

```

---

# 📋 Model Evaluation

The final model is evaluated using:

### Accuracy

Measures the overall percentage of correct predictions.

### Precision

Measures how many predicted anomalies were actually anomalies according to the reference labels.

### Recall

Measures how many reference anomalies were detected.

### F1 Score

Provides a balance between Precision and Recall.

### Confusion Matrix

The project also generates a confusion matrix:

```text
                 Predicted
              Normal  Anomaly

Actual Normal
Actual Anomaly

```

---

# 🚨 Anomaly Detection

After training the final model, anomalous users are extracted:

```python
anomalies = df[df["final_binary"] == 1]

```

The detected anomalies contain behavioral information such as:

```text
Daily Logins
Failed Password Attempts
Session Duration
Data Download
Off-Hours Activity
Unique IP Count

```

---

# 💾 Output

The detected anomalies are exported as:

```text
Anomalies.csv

```

This file can be used for:

- Security investigation
- User behavior analysis
- Further visualization
- Incident investigation
- Building downstream monitoring systems

---

# 📁 Project Structure

```text
User-Behavior-Anomaly-Detection/
│
├── User_Behavior_Anomaly_Detection_using_Isolation_Forest.ipynb
│
├── user_behavior_telemetry.csv
│
├── Anomalies.csv
│
├── Daily_Logins_distribution.png
├── Failed_Password_Attempts_distribution.png
├── Session_Duration_Min_distribution.png
├── Data_Download_MB_distribution.png
├── Off_Hours_Activity_Pct_distribution.png
├── Unique_IP_Count_distribution.png
│
├── reference_label_distribution.png
├── base_model_anomaly_score_distribution.png
├── base_model_confusion_matrix.png
├── final_model_anomaly_score_distribution.png
├── final_model_prediction_distribution.png
└── final_model_confusion_matrix.png

```

---

# 🔄 End-to-End ML Pipeline

```text
                    USER TELEMETRY
                         │
                         ▼
                  DATA COLLECTION
                         │
                         ▼
                  DATA INSPECTION
                         │
                         ▼
                  DATA CLEANING
                ┌────────┴────────┐
                │                 │
          Missing Values    Infinite Values
                │                 │
                └────────┬────────┘
                         ▼
                        EDA
                         │
                         ▼
              REFERENCE LABEL CREATION
                   (3-Sigma Rule)
                         │
                         ▼
                 FEATURE SELECTION
                         │
                         ▼
                  STANDARD SCALING
                         │
                         ▼
                ISOLATION FOREST
                         │
                         ▼
                  BASE PREDICTION
                         │
                         ▼
                  PARAMETER TUNING
                         │
                         ▼
                   FINAL MODEL
                         │
                         ▼
                 ANOMALY PREDICTION
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
           Normal                Anomaly
                                    │
                                    ▼
                              Anomalies.csv

```

---

# 🧠 Key Learning Outcomes

Through this project, I learned how to:

- Understand anomaly detection problems.
- Apply an unsupervised machine learning algorithm.
- Use Isolation Forest for behavioral anomaly detection.
- Handle missing and infinite values.
- Perform exploratory data analysis.
- Select meaningful behavioral features.
- Apply feature scaling.
- Understand Isolation Forest parameters.
- Generate anomaly scores.
- Tune model parameters manually.
- Evaluate anomaly detection results.
- Interpret confusion matrices.
- Export detected anomalies for further analysis.

---

# ⚠️ Important Project Limitation

This project uses **rule-based 3-sigma labels as reference labels** because the dataset does not contain verified attack/anomaly ground truth.

Therefore:

```text
Reference Label ≠ Real Security Ground Truth

```

The evaluation metrics indicate how closely Isolation Forest agrees with the defined statistical reference rule.

For a production security system, verified incident labels or expert-validated ground truth would provide a stronger basis for evaluation.

---

# 🔮 Future Improvements

Possible future improvements include:

- Real-world cybersecurity telemetry
- Verified attack labels
- Real-time anomaly detection
- Streaming user activity monitoring
- Alert generation
- Interactive Streamlit dashboard
- SHAP/explainability analysis
- Threshold optimization
- Model persistence using Joblib
- Automated monitoring pipeline
- Database integration
- API deployment using FastAPI

---

# 👨‍💻 Author

**Muhammad Aasik**

B.Sc Artificial Intelligence & Machine Learning Student

### Areas of Interest

- Python
- Backend Development
- Machine Learning
- Artificial Intelligence
- Data Science
- Anomaly Detection

---

# ⭐ Project Summary

**User Behavior Anomaly Detection using Isolation Forest** demonstrates an end-to-end unsupervised machine learning workflow for identifying unusual user behavior from telemetry data.

The project combines:

```text
Data Analysis
      +
Statistical Reference Rules
      +
Feature Engineering
      +
Feature Scaling
      +
Isolation Forest
      +
Parameter Tuning
      +
Model Evaluation
      +
Anomaly Export

```

to create a complete anomaly detection pipeline.