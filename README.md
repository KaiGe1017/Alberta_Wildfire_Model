# 🔥 Alberta Wildfire Prediction Model

Wildfires in Alberta pose a serious threat every year. This project uses public wildfire data to **analyze patterns and predict the final severity of wildfires** based on:

- 📅 Month
- 📍 Geographic location
- 🔥 Cause of wildfire
- 🌤️ Weather conditions

We hope this model can help relevant authorities **allocate resources more effectively** and respond in a timely manner.

---

### 📁 File Structure & Explanation

## 📘 0. Background & Case Study Folder
Contains public case studies and documents to:
- Understand the **context** of Alberta wildfires
- Justify why **data analysis and modeling** is necessary

---

## 📊 1. Datasets

- `historic_data.csv` / `historic_data.xlsx`  
  📌 **Raw data** from the Alberta Government’s open data portal  
  ⏳ Covers wildfire records from **2006–2023**

- `wildfire_data_dictionary.pdf`  
  📌 Official **data dictionary** from the public dataset website

- `Data_prepration_cleansing_SQLquery.sql`  
  📌 SQL scripts used for **data cleansing and import** to Power BI - I did the same process in .py file in modeling, if you don't do the Powerbi dashborad, could skip it.

- `Wildfire_data_dictionary.xlsx`  
  📌 **Authored by this project**  
  📄 Includes:
  - Cleaned and derived fields
  - A list of features used for modeling

---

## 📊 2. Data Analysis

This section includes exploratory visualizations and dashboard reports designed to answer key business questions such as:

- Which regions are most susceptible to wildfires?
- Do the causes of wildfires differ significantly across regions?
- How do weather conditions (e.g., temperature, humidity) influence wildfire size?

📁 `2.data analysis/`
- `Wildfire_Dashboard and Report.pptx`  
  👉 A Power BI dashboard/report addressing the business questions through visual insights. Ideal for business presentation use.

---

## 🤖 3. Modeling & Prediction

This section includes all modeling-related notebooks, trained models, and prediction outputs. Both **LightGBM** and **XGBoost** algorithms are used and compared for classification performance.

📁 `3.model/`

### 🧪 Notebooks
- `1.wildfire_size_prediction_preparation.ipynb`  
- `2.wildfire_size_prediction_model.ipynb`  
  ➤ Data preprocessing, feature engineering, and model training

- `3.wildfire_size_forecast_lightgbm.ipynb`  
- `3.wildfire_size_forecast_xgboost.ipynb`  
  ➤ Batch prediction using trained models on new user input

### 📦 Saved Model Files

| File Name            | Description                          |
|----------------------|--------------------------------------|
| `xgboost_model.pkl`  | Trained XGBoost model                |
| `lgbm_model.pkl`     | Trained LightGBM model               |
| `scaler.pkl`         | StandardScaler used for normalization |
| `imputer.pkl`        | Imputer used for filling missing values |
| `label_encoders.pkl` | Encoders for categorical variables   |

### 📄 Input & Output Samples

| File Name                       | Description                               |
|----------------------------------|-------------------------------------------|
| `user_input.xlsx`               | Sample user input file (month, location, weather, etc.) |
| `prediction_results_xgboost.xlsx` | Prediction results using XGBoost model   |
| `prediction_results_lightgbm.xlsx` | Prediction results using LightGBM model |

---

### 🔧 Python Libraries Used

| Library        | Description                                         |
|----------------|-----------------------------------------------------|
| `pandas`       | Data loading, transformation, and manipulation      |
| `numpy`        | Numerical computations and array operations         |
| `matplotlib`   | Visualization of charts and graphs                  |
| `seaborn`      | Statistical data visualization                      |
| `scikit-learn` | Preprocessing, model training, evaluation utilities |
| `imblearn`     | SMOTE: oversampling for class imbalance             |
| `lightgbm`     | LightGBM classifier (fast gradient boosting)        |
| `xgboost`      | XGBoost classifier for tree-based modeling          |
| `joblib`       | Saving/loading models and encoders                  |
| `openpyxl`     | Reading/writing Excel files                         |
| `pycaret`      | AutoML for model comparison and tuning              |
| `flask`        | Building a lightweight prediction web interface     |

or Install all dependencies with:

```bash
pip install -r requirements.txt
```

---



🧠 *If you use this project, please remember to **cite the original repository**. Attribution matters!* ✅

