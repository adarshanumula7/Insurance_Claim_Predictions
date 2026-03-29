# Porto Seguro Safe Driver Prediction
## Exploratory Data Analysis (EDA)

---

## Dataset Overview

- The dataset contains **595,212 rows** and **58 columns**.
- Feature naming convention:
    - Columns ending with **`_cat`** &rarr; categorical features
    - Columns ending with **`_bin`** &rarr; binary features
    - Remaining columns &rarr; numeric (continuous or ordinal)
- Missing values are encoded as **`-1`**, which were converted to `NAN` for analysis.

---

## 1. Missing Values Analysis

Features were first separated into:
- **Category columns**
- **Binary columns**
- **Numeric columns**

### 1.1 Missing Value Summary

| Feature      | Missing Count | Missing % |
|--------------|---------------|-----------|
|ps_car_03_cat |     411,231   | 69%       |
|ps_car_05_cat |     266,551   | 44.7%     |
|ps_reg_03     |     107,772   | 18%       |
|ps_car_14     |     42,620    | 7.16%     |
|ps_car_07_cat |     11,489    | 1.93%     |
|ps_ind_05_cat |     5809      | 0.97%     |
|ps_car_09_cat |	 569	   | 0.095%    |
|ps_ind_02_cat |	 216	   | 0.036%    |
|ps_car_01_cat |	 107	   | 0.018%    |
|ps_ind_04_cat |	 83	       | 0.014%    |
|ps_car_02_cat |	 5	       | 0.00084%  |
|ps_car_11	   |     5	       | 0.00084%  |
|ps_car_12	   |     1         | 0.00017%  |

---

### 1.2 Categorical Cols

- Columns with **< 1% missing values**:
    -  `ps_ind_02_cat`, `ps_ind_04_cat`, `ps_car_01_cat`,
    -   `ps_car_02_cat`, `ps_car_09_cat`, `ps_ind_05_cat`.
- **Action**: Impute with **mode**

- `ps_car_07_cat` (~2% missing):
    -  **Action**: Treat missing values as a **new category**.

- `ps_car_05_cat` contains 266,551(~45%) missing: 
- `ps_car_03_cat` contains 411,231(~70%) missing:
    - **Action**: Drop these features due to excessive missingness.

---

### 1.3 Binary cols

- No missing values present
- **No imputation required**

---

### 1.4 Numeric Cols

- Minor missing values:
    - `ps_car_11`, `ps_car_12` &rarr; negligible missing values
- Moderate missing values:  
    - `ps_reg_03` contains 107,772(~18%) missing values.
    - `ps_car_14` contains 42,620(~7%) missing values.

- **Imputation Strategy.**
- Use **median imputation** (robust to skewness and outliers)

---

## 2. Distribution

### 2.1 Numeric Columns

#### 2.1.1 Histplots

**Key Observations:**
- Strong **class imbalance** (majority target = 0)
- Significant overlap between target classes &rarr; weak linear separability

##### `ps_ind_*` Features
- `ps_ind_01`, `ps_ind_03`, `ps_ind_14`, `ps_ind_15` have **descrete values**.
- These are **ordinal features**, not continuous
- `ps_ind_14` is almost always zero &rarr; **low information content**

##### `ps_reg_*` Features
- True **continuous variables**.
- strong **right skew**.
- Potential candidates for transformation in linear models

##### `ps_car_*` Features
- `ps_car_11` &rarr; Ordinal, categorical in nature. 
- Remaining `ps_car_*` features &rarr; continuous

##### `ps_calc_*` Features
- Uniform and discrete patterns.
- Appear artificially engineered.
- Often weak predictors in practice.

---

#### 2.1.2 Boxplots

Purpose:
- Visualize spread
- Identify outliers

Features with heavy right tails:
- `ps_reg_03`
- `ps_car_12`
- `ps_car_13`
- `ps_car_14`

**Inference**:
- Presence of extreme values
- IQR-based outlier handling preferred over z-score.

---

#### 2.1.3 Skewness

Skewness formula:
\[
\text{Skewness} = \frac{E[(X - \mu)^3]}{\sigma^3}
\]

- Measures **asymmetry** of distribution
- Many numeric features are **positively skewed**

---

#### 2.1.4 Kurtosis

- Measures **tail heaviness**
- Several numeric features show **high kurtosis**
- Indicates presence of extreme values
- Tree-based models are robust to this behavior

---

### 2.2 Categorical Columns

Analysis using `value_counts(normalize=True)` revealed:
- Dominant category
- Rare category
- Near-constant features

#### Features with overly dominant categories

#### `ps_car_02_cat`
| Category      | percentage |
|---------------|------------|
| 1             |    83%     |
| 0             |    17%     |

#### `ps_car_07_cat`
| Category      | percentage |
|---------------|------------|
| 1             |    94%     |
| 0             |     6%     |

#### `ps_car_08_cat`
| Category      | percentage |
|---------------|------------|
| 1             |    83%     |
| 0             |    17%     |

#### `ps_car_10_cat`
| Category      | percentage |
|---------------|------------|
| 1             |    99%     |
| 0             |    0.8%    |
| 2             |    0.2%    |

#### `ps_ind_14`
| Category      | percentage |
|---------------|------------|
|0              |    99%     |
|1-4            |    01%     |

**Inference**:
- Near-constant features contribute little information
- Candidates for removal or low importance

---

## 3. Correlation Analysis
- use Pearson correlation (`df.corr()`)
- **No** numeric feature pairs showed strong linear correlation
- Multicollinearity is **not a major concern**

---

## 4. Class Imbalance

Target Distribution:
- `target = 0` &rarr; ~96% of samples
- `target = 1` &rarr; ~4% of samples

**Modeling Implications:**
- Accuracy is misleading
- Primary metrics: **AUC-ROC** and **Normalized Gini** (`2 × AUC - 1`)
- Use class weights / scale_pos_weight in boosting models

---

## Key Conclusions

### :one: Which Features Matter?

Likely informative:
- `ps_car_13`
- `ps_reg_03`
- `ps_reg_02`
- `ps_ind_15`

Moderately useful:
- Ordinal `ps_ind_*` features (except `ps_ind_14`)

---

### :two: Which Features are Redundant?

- `ps_car_03_cat` (69% missing)
- `ps_car_05_cat` (45% missing)
- `ps_calc_*` features (uniform, weak signal)
- `ps_ind_14` (near constant)

---

### :three: How to Preprocess?

- Convert `-1` &rarr; `NaN`
- Drop high-missing categorical features
- Median imputation for numeric features
- Mode / new category for categorical features
- Avoid aggressive outlier removal
- Encode:
    - Low cardinality &rarr; One-hot
    - High cardinality &rarr; Target encoding

 ---

 ### 4️⃣ Which Model is Suitable?

 :x: Logistic Regression (without heavy feature engineering)
 
 :x: Distance-based models

 ✅ **Tree-based models**:
 - LightGBM
 - XGBoost
 - CatBoost

Reasons:
- Handle skewness & kurtosis well
- Robust to outliers
- Handle missing values effectively
- Capture feature interactions

---

## Final Remark

The dataset shows weak individual feature separation but contains predictive signal through **nonlinear interactions**, making **boosting-based models** the most suitable choice.