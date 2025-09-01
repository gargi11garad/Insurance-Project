# 🏥 Insurance Charges Prediction 

This project focuses on analyzing and preparing the **Medical Insurance Dataset** for machine learning models.  
We perform **data cleaning, exploratory data analysis (EDA), and statistical tests** to identify significant features that influence medical insurance charges.

---

## 📌 Project Overview
The goal of this project is to:
1. Explore and preprocess the insurance dataset.
2. Perform **data cleaning and feature engineering**.
3. Conduct **exploratory data analysis (EDA)** to understand data distributions.
4. Apply **statistical tests (Chi-Square, Correlation, etc.)** for feature selection.
5. Prepare the dataset for predictive modeling (covered in the next phase).

---

## 📂 Dataset
The dataset contains the following columns:
- `age`: Age of the insured person  
- `sex`: Gender (male/female)  
- `bmi`: Body Mass Index  
- `children`: Number of children/dependents covered by health insurance  
- `smoker`: Smoking status (yes/no)  
- `region`: Residential area in the US (northeast, southeast, southwest, northwest)  
- `charges`: Medical insurance charges (target variable)  

---

## ⚙️ Steps Implemented
### 1. Data Loading & Cleaning
- Imported dataset using **Pandas**.
- Handled **missing values** and removed duplicates.
- Encoded categorical variables where necessary.

### 2. Exploratory Data Analysis (EDA)
- Visualized distributions of numerical features (`age`, `bmi`, `charges`).
- Count plots for categorical variables (`sex`, `smoker`, `region`).
- Boxplots to study effects of smoking, age, and region on charges.
- Outlier detection and handling.

### 3. Feature Engineering
- Created **binned target variable (`charges_bin`)** using quartiles for statistical testing.
- Normalized/standardized numeric features (optional).

### 4. Statistical Tests
- **Correlation analysis** for numerical features.  
- **Chi-Square test of independence** for categorical features vs. `charges_bin`:
  ```python
  from scipy.stats import chi2_contingency
  contingency = pd.crosstab(df['smoker'], df['charges_bin'])
  chi2_stat, p_val, _, _ = chi2_contingency(contingency)

