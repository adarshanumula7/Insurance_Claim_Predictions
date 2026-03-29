# Porto Seguro Safe Driver Prediction -- EDA Summary

## 🔴 First Global Observation (Very Important)

From the plots:

-   🔵 **Blue (target = 0)** dominates heavily\
-   🟠 **Orange (target = 1)** is very small

This confirms:

> ⚠️ **Severe class imbalance (≈ 3--4% positive class)**

### 🔎 Modeling Implications:

-   ❌ Accuracy is **not** a reliable metric
-   ✅ Use **AUC-ROC**
-   ✅ Use **Normalized Gini**
-   ✅ Use **F1-score**
-   ✅ Use **Precision--Recall**
-   ✅ Use **class weights or boosting models**

------------------------------------------------------------------------

# 1️⃣ ps_ind\_\* Features

## 🔹 ps_ind_01

-   Discrete values (0--7)
-   Highly imbalanced at 0 and 1
-   Long decreasing pattern

**Inference:** - Ordinal feature\
- Not continuous\
- No major separation between target classes\
- Likely weak individual predictor

------------------------------------------------------------------------

## 🔹 ps_ind_03

-   Discrete 0--11\
-   Bell-shaped discrete distribution\
-   Center around 3--6

**Inference:** - Ordinal categorical\
- Natural ordering present\
- Slight variation between target classes\
- May interact with other `ps_ind` features

------------------------------------------------------------------------

## 🔹 ps_ind_14

-   Almost all values are 0

**Inference:** - Extremely sparse\
- Low standalone predictive power\
- May matter in interaction

------------------------------------------------------------------------

## 🔹 ps_ind_15

-   Spread between 0--12\
-   Increasing then decreasing trend

**Inference:** - Strong ordinal structure\
- Small shift between classes\
- Likely moderate predictive importance

------------------------------------------------------------------------

# 2️⃣ ps_reg\_\* Features (Continuous)

## 🔹 ps_reg_01

-   Numeric fractions\
-   Concentrated near 0.7--0.9

**Inference:** - Right-heavy\
- May need scaling\
- Possible monotonic risk relationship

------------------------------------------------------------------------

## 🔹 ps_reg_02

-   Strong right skew\
-   Heavy mass near 0

**Inference:** - Highly skewed\
- Log transform may help\
- Likely nonlinear effect

------------------------------------------------------------------------

## 🔹 ps_reg_03

-   Gamma-like distribution\
-   Strong right skew

**Inference:** - Continuous variable\
- Long tail\
- Outliers present\
- Good candidate for log transformation

------------------------------------------------------------------------

# 3️⃣ ps_car\_\* Features

## 🔹 ps_car_11

-   Distinct spikes (0--3)\
-   Large spike at 3

**Inference:** - Categorical disguised as numeric\
- Do NOT treat as continuous\
- Prefer one-hot encoding

------------------------------------------------------------------------

## 🔹 ps_car_12

-   Narrow range (0.35--0.45)\
-   Slight bell shape

**Inference:** - Continuous\
- Likely normalized\
- May contain predictive signal

------------------------------------------------------------------------

## 🔹 ps_car_13

-   Clean right skew\
-   Smooth density

**Inference:** - Strong continuous feature\
- Often highly important in this dataset\
- Avoid blind binning

------------------------------------------------------------------------

## 🔹 ps_car_14

-   Tight concentration\
-   Small variance

**Inference:** - Low variance\
- Limited standalone power

------------------------------------------------------------------------

## 🔹 ps_car_15

-   Discrete increasing values\
-   Count-like structure

**Inference:** - Ordinal\
- Visible target difference\
- Possibly useful

------------------------------------------------------------------------

# 4️⃣ ps_calc\_\* Features

⚠️ In many Kaggle solutions, these were weak predictors.

## 🔹 ps_calc_01 / 02 / 03

-   Uniform discrete pattern\
-   No clear separation

**Inference:** - Very weak predictors\
- Likely noise

------------------------------------------------------------------------

## 🔹 ps_calc_04 to ps_calc_14

-   Symmetric engineered-looking distributions\
-   Evenly spaced values

**Inference:** - Likely synthetic features\
- Often low importance\
- Frequently dropped

------------------------------------------------------------------------

# 🎯 Likely Strongest Signals (Visual Inspection)

## ✅ Stronger Predictors

-   ps_car_13\
-   ps_reg_03\
-   ps_reg_02\
-   ps_ind_15

## ❌ Likely Weak Predictors

-   ps_calc\_\* features\
-   ps_ind_14\
-   ps_car_14
