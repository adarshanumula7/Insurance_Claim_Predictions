# Porto Seguro Safe Driver Prediction

**Complete From-Scratch Machine Learning Solution**  
**Decision Tree • Random Forest • XGBoost • LightGBM (all built from scratch)**

[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Project Overview

This repository contains a **full end-to-end** solution for the **Porto Seguro Safe Driver Prediction** competition (Kaggle).

**Goal**: Predict the probability that a driver will file an insurance claim in the next year.

**Key Highlights**:
- Severe class imbalance (~3.65% positive class)
- Every core model built **from scratch** (no black-box libraries for the algorithms)
- Every hyper-parameter is exposed and documented
- Full EDA, preprocessing, and feature engineering
- Clean, reproducible, production-ready code

---


## 📊 DataSet (Porto Seguro Safe Driver Prediction)

- This Dataset has **595k** records.
- Null values are indicated by value **`-1`**.
- Types of Columns:
  - Categorical columns ends with **'_cat'**.
  - Binary Columns ends with **'_bin'**.
  - Every thing else are Numeric columns.
- **Severe imbalance**: 96.35% `target=0`, 3.65% `target=1`

---

## 📁 Project Structure

--------------------------------
```bash
Porto-Seguro-Safe-Driver/
├── Raw-Dataset/
│   ├── train.csv
│   └── test.csv
├── Cleaned-Dataset/
│   ├── test_2.csv    # Not imputed category null values for LightGBM
│   ├── test.csv
│   ├── train_2.csv   # Not imputed category null values for LightGBM
│   └── train.csv
├── EDA/
│   ├── EDA.ipynb
│   ├── EDA.md
│   ├── Histplot_analysis.md
│   └── Histplot_numeric_cols.png
├── Models/
│   ├── Feature_Engineering.ipynb
│   ├── Decision_Tree.ipynb
│   ├── Random_Forest.ipynb
│   ├── XGBoost.ipynb
│   └── LightGBM.ipynb
├── Final-Submission/
│   ├── DT-submission_best.csv
│   ├── RF-submission_best.csv
│   ├── XGBoost-submission_best.csv
│   └── LightGBM-submission_best.csv
├── requirements.md
├── README.md          ← You are here
└── requirements.txt
--------------------------------

---

# 🔍 Exploratory Data Analysis (EDA)

Detailed EDA is available in:

- `EDA.md`
- `Histplot_analysis.md`

## Key Findings

- High missing values → `ps_car_03_cat (69%)`, `ps_car_05_cat (44.7%)` → dropped  
- Strong right-skew in numeric features  
- Many near-constant / dominant-category features  
- `ps_calc_*` features are weak / engineered noise  
- Strongest signals: `ps_car_13`, `ps_reg_03`, `ps_reg_02`, `ps_ind_15`  
- Primary metrics: **AUC-ROC** and **Normalized Gini (2 × AUC - 1)**  


---

# 🛠️ Preprocessing & Feature Engineering

Pipeline: `02_Feature_Engineering.ipynb`

## Steps

- Convert `-1 → NaN`
- Drop high-missing columns:
  - `ps_car_03_cat`
  - `ps_car_05_cat`
- Treat missing in `ps_car_07_cat` as a new category  
- Median imputation for numeric columns  
- Mode imputation for low-missing categorical columns  
- Remove near-constant features:
  - `ps_car_10_cat`
  - `ps_ind_14`

## Encoding

- Target Encoding → high-cardinality (`ps_car_11_cat`)  
- One-Hot Encoding → low-cardinality categorical features  

> No aggressive outlier removal (tree models are robust)

Cleaned datasets are saved in: `Cleaned-Dataset/`


---

# 🧪 Models Built From Scratch

All models expose every single hyper-parameter for complete transparency.

## Model Comparison

| Model Type | Implementation | Key Parameters Exposed | Best Gini (Val) |
|-----------|--------------|----------------------|----------------|
| Decision Tree | From scratch | `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `pos_weight` | ~0.2122 |
| Random Forest | From scratch | `n_estimators`, `max_depth`, `min_samples_split`, `max_features`, `bootstrap`, `OOB` | 0.249 |
| XGBoost-style GBM | From scratch | `n_estimators`, `learning_rate`, `max_depth`, `min_child_weight`, `colsample_bytree`, `subsample`, `reg_alpha`, `reg_lambda`, `reg_gamma`, `scale_pos_weight` | ~0.258 |
| LightGBM-style GBM | From scratch | `num_iterations`, `num_leaves`, `min_data_in_leaf`, `min_sum_hessian_in_leaf`, `feature_fraction`, `bagging_fraction`, `reg_alpha`, `reg_lambda`, `GOSS` | 0.234 |
| LightGBM-style GBM | From scikit-learn | `num_iterations`, `num_leaves`, `min_data_in_leaf`, `min_sum_hessian_in_leaf`, `feature_fraction`, `bagging_fraction`, `reg_alpha`, `reg_lambda`, `GOSS` | 0.268 |

### Included in All Notebooks

- Detailed comments on every parameter  
- Training logs (AUC per tree)  
- Validation metrics  


---

# 📈 Results Summary

| Model | Validation AUC | Gini | Comment |
|------|---------------|------|---------|
| Decision Tree (scratch) | ~0.588 | ~0.176 | Strong baseline |
| Random Forest (scratch) | 0.624 | 0.249 | Best from-scratch ensemble |
| XGBoost (scratch) | ~0.629 | ~0.258 | Excellent non-linear capture |
| LightGBM (scratch) | ~0.617 | ~0.234 | Leaf-wise growth |
| LightGBM (sklearn) | ~0.634 | ~0.268 | Leaf-wise growth |

---

# 🚀 How to Reproduce

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd Porto-Seguro-Safe-Driver

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create below folders in your repo.
1. Raw-Dataset (Store raw Dataset test.csv, train.csv here)
2. Cleaned-Dataset
3. Final-Submission

# 4. Run notebooks in order
jupyter notebook EDA/EDA.ipynb
jupyter notebook Models/feature_engineering.ipynb
# Continue with model notebooks...
```

> Note: Core algorithms use only **numpy + pandas**.  
> Scikit-learn is used only for comparison and encoding helpers.


---

# 📦 Dependencies

See `requirements.md` for full list.

## Main Packages

- pandas  
- numpy  
- matplotlib  
- seaborn  
- category_encoders  
- scikit-learn (only for metrics & comparison)  


---

# 📄 Final Submissions

Located in `Final-Submission/`:

- `DT-submission_best.csv`  
- `RF-submission_best.csv` 
- `XGBoost-submission_best`   
- `LightGBM-submission_best` 

Ready for Kaggle submission.


---

# 🎯 Learning Outcomes

- Implemented gradient boosting mathematics (first & second-order derivatives)  
- Built:
  - Leaf-wise growth  
  - GOSS  
  - Exact greedy splits  
  - L1/L2 regularization  
- Deep understanding of every hyper-parameter in XGBoost and LightGBM  
- Handled real-world challenges:
  - Missing values  
  - Severe imbalance  
  - High cardinality  


---

# 👤 Author

**Adarsh**  
Hyderabad, Telangana, India  

Built as a complete **"from-scratch" ML journey** to deeply understand modern gradient boosting algorithms.
