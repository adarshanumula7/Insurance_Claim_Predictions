# Insurance Claim prediction

- In this project we train models to predict insurance claim on a dataset with 595k records.
  

## 1. DataSet (Porto Seguro Safe Driver Prediction)

- This Dataset has 595k records.
- Null values are indicated by value -1.
- Types of Columns:
  - Categorical columns ends with '_cat'.
  - Binary Columns ends with '_bin'.
  - Every thing else are Numeric columns.
  

## 2. Project Structure

--------------------------------
- Raw-Dataset/train.csv
          - / test.csv
- EDA
- Models
--------------------------------

## 3. EDA

### 1. Analyze Missing Values

#### Categorical Cols

- ['ps_ind_02_cat', 'ps_ind_04_cat', 'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_09_cat', 'ps_ind_05_cat', 'ps_car_07_cat'] => **Consider as seperate category**

- For highly sparse cols => ['ps_car_05_cat', 'ps_car_03_cat'] => drop columns

#### Binary Columns

- No Missing values.

#### Numeric Columns

- less sparse columns => ['ps_car_11', 'ps_car_12'] => Impute with median (Robust to Outliers)
- highly sparse Columns => ['ps_reg_03', 'ps_car_14'] => Consider using missing binary column if needed, and Impute with median.

### 2. Distributions



### 3. Correlation

### 4. Class imbalance

## 4. Data Preprocessing

### 1. Handle missing

### 2. Encoding

### 3. Scaling (if required)

### 4. Optional transformations
