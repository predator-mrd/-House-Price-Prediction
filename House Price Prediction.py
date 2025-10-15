"""
HOUSE PRICE PREDICTION MODEL - 90%+ ACCURACY TARGET
Complete Machine Learning Project for Google Colab
Dataset: 13,603 training samples, 6,700 test samples
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')

print("="*70)
print("HOUSE PRICE PREDICTION MODEL")
print("Target: 90%+ R² Score")
print("="*70)

# Upload files
from google.colab import files
print("\n📂 Upload your CSV files:")
print("1. Upload df_train.csv")
uploaded = files.upload()
print("2. Upload df_test.csv")
uploaded = files.upload()

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n" + "="*70)
print("STEP 1: DATA LOADING")
print("="*70)

df_train = pd.read_csv("/df_train.csv")
df_test = pd.read_csv("/df_test.csv")

print(f"\n✓ Training data: {df_train.shape[0]} samples, {df_train.shape[1]} features")
print(f"✓ Test data: {df_test.shape[0]} samples, {df_test.shape[1]} features")

# ============================================================================
# STEP 2: DATA PREPROCESSING
# ============================================================================
print("\n" + "="*70)
print("STEP 2: DATA PREPROCESSING")
print("="*70)

# Drop date column (not useful for prediction)
if 'date' in df_train.columns:
    df_train = df_train.drop('date', axis=1)
    df_test = df_test.drop('date', axis=1)
    print("   ✓ Dropped 'date' column")

# Convert boolean columns to integers
bool_columns = df_train.select_dtypes(include=['bool']).columns
for col in bool_columns:
    df_train[col] = df_train[col].astype(int)
    df_test[col] = df_test[col].astype(int)
print(f"   ✓ Converted {len(bool_columns)} boolean columns to numeric")

print(f"\n   No missing values in training: {df_train.isnull().sum().sum() == 0}")
print(f"   No missing values in test: {df_test.isnull().sum().sum() == 0}")

# ============================================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================================
print("\n" + "="*70)
print("STEP 3: ADVANCED FEATURE ENGINEERING")
print("="*70)

def create_features(df):
    """Create advanced features for better predictions"""
    df = df.copy()
    
    # Polynomial features for living area
    df['living_squared'] = df['living_in_m2'] ** 2
    df['living_cubed'] = df['living_in_m2'] ** 3
    
    # Area per bedroom
    df['area_per_bedroom'] = df['living_in_m2'] / (df['bedrooms'] + 1)
    
    # Bathroom features
    df['total_bathrooms'] = df['real_bathrooms'] + df['has_lavatory']
    df['bathroom_bedroom_ratio'] = df['real_bathrooms'] / (df['bedrooms'] + 1)
    
    # Quality indicators
    df['quality_score'] = (df['grade'] * 2 + 
                           df['perfect_condition'] * 3 + 
                           df['nice_view'] * 2 + 
                           df['renovated'] * 2)
    
    # Interaction features
    df['grade_x_living'] = df['grade'] * df['living_in_m2']
    df['bedrooms_x_bathrooms'] = df['bedrooms'] * df['real_bathrooms']
    df['zone_x_living'] = df['quartile_zone'] * df['living_in_m2']
    df['zone_x_grade'] = df['quartile_zone'] * df['grade']
    
    # Luxury indicator
    df['is_luxury'] = ((df['grade'] >= 4) & 
                       (df['living_in_m2'] > 200) & 
                       (df['nice_view'] == 1)).astype(int)
    
    # Seasonal features (sine/cosine for cyclical nature)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df

# Apply feature engineering
df_train = create_features(df_train)
df_test = create_features(df_test)

print(f"   ✓ Created {len(df_train.columns) - 13} new features")
print(f"   ✓ Total features: {len(df_train.columns) - 1}")

# ============================================================================
# STEP 4: EXPLORATORY DATA ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("STEP 4: EXPLORATORY DATA ANALYSIS")
print("="*70)

# Price statistics
print(f"\n📊 Price Statistics:")
print(f"   Min: ${df_train['price'].min():,.0f}")
print(f"   Max: ${df_train['price'].max():,.0f}")
print(f"   Mean: ${df_train['price'].mean():,.0f}")
print(f"   Median: ${df_train['price'].median():,.0f}")
print(f"   Std: ${df_train['price'].std():,.0f}")

# Visualization 1: Price Distribution
print("\n📊 Visualization 1: Price Distribution")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Original price
ax1.hist(df_train['price'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
ax1.axvline(df_train['price'].mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean: ${df_train["price"].mean():,.0f}')
ax1.axvline(df_train['price'].median(), color='green', linestyle='--', linewidth=2,
           label=f'Median: ${df_train["price"].median():,.0f}')
ax1.set_xlabel('Price ($)', fontweight='bold', fontsize=12)
ax1.set_ylabel('Frequency', fontweight='bold', fontsize=12)
ax1.set_title('House Price Distribution', fontweight='bold', fontsize=14)
ax1.legend()
ax1.grid(alpha=0.3)

# Log-transformed price
ax2.hist(np.log1p(df_train['price']), bins=50, color='coral', edgecolor='black', alpha=0.7)
ax2.set_xlabel('Log(Price)', fontweight='bold', fontsize=12)
ax2.set_ylabel('Frequency', fontweight='bold', fontsize=12)
ax2.set_title('Log-Transformed Price Distribution', fontweight='bold', fontsize=14)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Visualization 2: Feature Correlations
print("\n📊 Visualization 2: Top Feature Correlations with Price")

X_temp = df_train.drop('price', axis=1)
correlations = X_temp.corrwith(df_train['price']).sort_values(ascending=False)

plt.figure(figsize=(12, 8))
top_15 = correlations.head(15)
colors = ['green' if x > 0.5 else 'orange' if x > 0.3 else 'coral' for x in top_15]

plt.barh(range(len(top_15)), top_15.values, color=colors, edgecolor='black', alpha=0.8)
plt.yticks(range(len(top_15)), top_15.index)
plt.xlabel('Correlation with Price', fontweight='bold', fontsize=12)
plt.title('Top 15 Features Correlated with Price', fontweight='bold', fontsize=14)
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

print("\nTop 10 Features:")
for i, (feat, corr) in enumerate(correlations.head(10).items(), 1):
    print(f"   {i:2d}. {feat:25s}: {corr:.4f}")

# ============================================================================
# STEP 5: PREPARE DATA FOR MODELING
# ============================================================================
print("\n" + "="*70)
print("STEP 5: DATA PREPARATION")
print("="*70)

# Separate features and target
X_train_full = df_train.drop('price', axis=1)
y_train_full = df_train['price']
X_test_full = df_test.drop('price', axis=1)
y_test_full = df_test['price']

print(f"\n   Training features: {X_train_full.shape}")
print(f"   Training target: {y_train_full.shape}")
print(f"   Test features: {X_test_full.shape}")
print(f"   Test target: {y_test_full.shape}")

# ============================================================================
# STEP 6: TRAIN MULTIPLE MODELS
# ============================================================================
print("\n" + "="*70)
print("STEP 6: MODEL TRAINING & COMPARISON")
print("="*70)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge (alpha=10)": Ridge(alpha=10),
    "Lasso (alpha=100)": Lasso(alpha=100, max_iter=5000),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, 
        max_depth=25, 
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.1,
        random_state=42
    )
}

results = []

print("\n⏳ Training models...\n")

for name, model in models.items():
    print(f"   Training {name}...")
    
    # Train
    model.fit(X_train_full, y_train_full)
    
    # Predict
    y_train_pred = model.predict(X_train_full)
    y_test_pred = model.predict(X_test_full)
    
    # Metrics
    train_r2 = r2_score(y_train_full, y_train_pred)
    test_r2 = r2_score(y_test_full, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train_full, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test_full, y_test_pred))
    mae = mean_absolute_error(y_test_full, y_test_pred)
    
    results.append({
        "Model": name,
        "Train_R2": train_r2,
        "Test_R2": test_r2,
        "Overfit_Gap": train_r2 - test_r2,
        "Test_RMSE": test_rmse,
        "MAE": mae
    })
    
    print(f"      Train R²: {train_r2:.4f}")
    print(f"      Test R²: {test_r2:.4f}")
    print(f"      RMSE: ${test_rmse:,.0f}\n")

# ============================================================================
# STEP 7: RESULTS COMPARISON
# ============================================================================
print("=" * 70)
print("MODEL COMPARISON RESULTS")
print("=" * 70)

results_df = pd.DataFrame(results).sort_values('Test_R2', ascending=False)
print("\n" + results_df.to_string(index=False))

best_model_name = results_df.iloc[0]['Model']
best_r2 = results_df.iloc[0]['Test_R2']
best_rmse = results_df.iloc[0]['Test_RMSE']
best_model = models[best_model_name]

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   Test R² Score: {best_r2:.4f} ({best_r2*100:.2f}%)")
print(f"   Test RMSE: ${best_rmse:,.0f}")

if best_r2 >= 0.90:
    print(f"\n✅ 🎉 SUCCESS! Achieved {best_r2*100:.2f}% accuracy!")
elif best_r2 >= 0.85:
    print(f"\n✅ EXCELLENT! {best_r2*100:.2f}% accuracy - very close to 90%!")
else:
    print(f"\n✓ GOOD! {best_r2*100:.2f}% accuracy achieved")

# ============================================================================
# STEP 8: VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 70)
print("DETAILED VISUALIZATIONS")
print("=" * 70)

# Get best model predictions
y_pred = best_model.predict(X_test_full)

# Visualization 3: Model Comparison
print("\n📊 Visualization 3: Model Performance Comparison")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

models_list = results_df['Model'].tolist()
test_r2_vals = results_df['Test_R2'].values

# R² comparison
bars = ax1.bar(range(len(models_list)), test_r2_vals, 
              color='steelblue', edgecolor='black', alpha=0.8)
ax1.set_xticks(range(len(models_list)))
ax1.set_xticklabels(models_list, rotation=45, ha='right')
ax1.set_ylabel('R² Score', fontweight='bold', fontsize=12)
ax1.set_title('Model R² Score Comparison', fontweight='bold', fontsize=14)
ax1.axhline(y=0.9, color='red', linestyle='--', linewidth=2, label='90% Target')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, test_r2_vals):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9)

# RMSE comparison
test_rmse_vals = results_df['Test_RMSE'].values
bars2 = ax2.barh(models_list, test_rmse_vals, color='coral', edgecolor='black', alpha=0.8)
ax2.set_xlabel('RMSE ($)', fontweight='bold', fontsize=12)
ax2.set_title('Model RMSE Comparison', fontweight='bold', fontsize=14)
ax2.invert_yaxis()
ax2.grid(axis='x', alpha=0.3)

for bar, val in zip(bars2, test_rmse_vals):
    ax2.text(val, bar.get_y() + bar.get_height()/2,
            f'${val:,.0f}', va='center', fontsize=9)

plt.tight_layout()
plt.show()

# Visualization 4: Actual vs Predicted
print("\n📊 Visualization 4: Actual vs Predicted Prices")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Scatter plot
ax1.scatter(y_test_full, y_pred, alpha=0.5, s=30, color='steelblue', edgecolor='black', linewidth=0.5)
ax1.plot([y_test_full.min(), y_test_full.max()], [y_test_full.min(), y_test_full.max()],
         '--', color='red', linewidth=2, label='Perfect Prediction')
ax1.set_xlabel('Actual Price ($)', fontweight='bold', fontsize=12)
ax1.set_ylabel('Predicted Price ($)', fontweight='bold', fontsize=12)
ax1.set_title(f'{best_model_name}: Actual vs Predicted', fontweight='bold', fontsize=14)
ax1.legend()
ax1.grid(alpha=0.3)

r2_text = f'R² = {best_r2:.4f}\nAccuracy: {best_r2*100:.2f}%'
ax1.text(0.05, 0.95, r2_text, transform=ax1.transAxes,
         fontsize=12, fontweight='bold', va='top',
         bbox=dict(boxstyle='round', facecolor='lightgreen' if best_r2 >= 0.9 else 'yellow', alpha=0.9))

# Residual plot
residuals = y_test_full - y_pred
ax2.scatter(y_pred, residuals, alpha=0.5, s=30, color='coral', edgecolor='black', linewidth=0.5)
ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax2.set_xlabel('Predicted Price ($)', fontweight='bold', fontsize=12)
ax2.set_ylabel('Residuals ($)', fontweight='bold', fontsize=12)
ax2.set_title('Residual Plot', fontweight='bold', fontsize=14)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Visualization 5: Error Distribution
print("\n📊 Visualization 5: Prediction Error Analysis")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

errors = y_pred - y_test_full
ax1.hist(errors, bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
ax1.axvline(errors.mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean: ${errors.mean():,.0f}')
ax1.axvline(0, color='green', linestyle='--', linewidth=2, label='Zero Error')
ax1.set_xlabel('Prediction Error ($)', fontweight='bold', fontsize=12)
ax1.set_ylabel('Frequency', fontweight='bold', fontsize=12)
ax1.set_title('Distribution of Errors', fontweight='bold', fontsize=14)
ax1.legend()
ax1.grid(alpha=0.3)

percentage_errors = (errors / y_test_full) * 100
ax2.hist(percentage_errors, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
ax2.axvline(percentage_errors.mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean: {percentage_errors.mean():.2f}%')
ax2.axvline(0, color='green', linestyle='--', linewidth=2)
ax2.set_xlabel('Percentage Error (%)', fontweight='bold', fontsize=12)
ax2.set_ylabel('Frequency', fontweight='bold', fontsize=12)
ax2.set_title('Percentage Error Distribution', fontweight='bold', fontsize=14)
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# STEP 9: EXPORT RESULTS
# ============================================================================
print("\n" + "=" * 70)
print("EXPORTING RESULTS")
print("=" * 70)

try:
    # Model comparison
    results_df.to_csv("model_comparison.csv", index=False)
    print("   ✓ Saved: model_comparison.csv")
    
    # Predictions
    predictions_df = pd.DataFrame({
        'Actual_Price': y_test_full.values,
        'Predicted_Price': y_pred,
        'Error': errors.values,
        'Percentage_Error': percentage_errors.values,
        'Absolute_Error': np.abs(errors.values)
    })
    predictions_df.to_csv("predictions.csv", index=False)
    print("   ✓ Saved: predictions.csv")
    
except Exception as e:
    print(f"   ⚠️ Export warning: {e}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("✅ HOUSE PRICE PREDICTION - COMPLETE!")
print("=" * 70)

print(f"\n📊 FINAL RESULTS:")
print(f"   • Best Model: {best_model_name}")
print(f"   • Test R² Score: {best_r2:.4f} ({best_r2*100:.2f}% Accuracy)")
print(f"   • Test RMSE: ${best_rmse:,.0f}")
print(f"   • Mean Absolute Error: ${results_df.iloc[0]['MAE']:,.0f}")
print(f"   • Average Percentage Error: {np.mean(np.abs(percentage_errors)):.2f}%")

print(f"\n📁 FILES CREATED:")
print("   • model_comparison.csv")
print("   • predictions.csv")

print(f"\n📊 DATASET SUMMARY:")
print(f"   • Training samples: {len(X_train_full):,}")
print(f"   • Test samples: {len(X_test_full):,}")
print(f"   • Total features: {X_train_full.shape[1]}")

print(f"\n🎯 MODEL INSIGHTS:")
print(f"   • The model explains {best_r2*100:.2f}% of price variance")
print(f"   • Average prediction error: ${results_df.iloc[0]['MAE']:,.0f}")
print(f"   • Overfitting gap: {results_df.iloc[0]['Overfit_Gap']:.4f}")

if best_r2 >= 0.90:
    print(f"\n🏆 ACHIEVEMENT UNLOCKED: 90%+ ACCURACY!")

print("\n" + "=" * 70)
