# 🏠 House Price Prediction

This project builds and compares multiple regression models to predict house prices using a clean numeric dataset. It includes data preprocessing, feature scaling, log transformation, model evaluation, and visualization.

## 📁 Dataset

The dataset used contains various numeric features related to houses (e.g., square footage, number of rooms, etc.) and the target variable `Price`.

> **Note:** The dataset (`data_house.csv`) is not included in this repository. Please add your own or modify the code accordingly.

---

## 📊 Models Used

- **Linear Regression**
- **Ridge Regression**
- **Lasso Regression**
- **Random Forest Regressor**

---

## 🛠️ Workflow

1. **Data Cleaning**  
   - Removed non-numeric and missing-value columns.
   - Applied log transformation to target (`Price`) to reduce skewness.

2. **Train-Test Split**  
   - Split the dataset into 80% training and 20% testing sets.

3. **Feature Scaling**  
   - Standardized features using `StandardScaler`.

4. **Model Training & Evaluation**  
   - Trained and evaluated four models using RMSE and R² score.

5. **Visualization**  
   - Plotted actual vs. predicted prices for the best-performing model.

---

## 📈 Results Summary

| Model             | RMSE    | R² Score |
|------------------|---------|----------|
| Linear Regression| XX.XX   | 0.XXX    |
| Ridge            | XX.XX   | 0.XXX    |
| Lasso            | XX.XX   | 0.XXX    |
| Random Forest    | XX.XX   | 0.XXX    |

> *(These will be filled in after running the notebook with your dataset)*

---

## 📦 Requirements

Install dependencies using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
