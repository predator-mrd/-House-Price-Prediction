"""
ALGERIAN FOREST FIRES PREDICTION - LINEAR REGRESSION PROJECT
TARGET: FWI (Fire Weather Index) Prediction with 90%+ R² Score
OPTIMIZED FOR GOOGLE COLAB WITH INLINE VISUALIZATIONS
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')

print("="*70)
print("ALGERIAN FOREST FIRES - FWI PREDICTION")
print("Linear Regression | Target: 90%+ R² Score")
print("="*70)

# Upload dataset
from google.colab import files
print("\n📂 Upload your Algerian_forest_fires_cleaned_dataset.csv:")
uploaded = files.upload()

# ============================================================================
# STEP 1: LOAD AND EXPLORE DATA
# ============================================================================
print("\n" + "="*70)
print("STEP 1: DATA LOADING & EXPLORATION")
print("="*70)

df = pd.read_csv("/Algerian_forest_fires_cleaned_dataset.csv")
print(f"\n✓ Loaded {len(df)} samples with {len(df.columns)} columns")

print("\nDataset Overview:")
print(f"   Rows: {df.shape[0]}")
print(f"   Columns: {df.shape[1]}")
print(f"   Features: {list(df.columns)}")

print("\nTarget Variable (FWI - Fire Weather Index):")
print(f"   Min: {df['FWI'].min():.2f}")
print(f"   Max: {df['FWI'].max():.2f}")
print(f"   Mean: {df['FWI'].mean():.2f}")
print(f"   Median: {df['FWI'].median():.2f}")

print("\nMissing Values:")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("   ✓ No missing values!")
else:
    print(missing[missing > 0])

# ============================================================================
# STEP 2: DATA PREPROCESSING
# ============================================================================
print("\n" + "="*70)
print("STEP 2: DATA PREPROCESSING")
print("="*70)

# Drop year (all same value)
if 'year' in df.columns and df['year'].nunique() == 1:
    df = df.drop('year', axis=1)
    print("   ✓ Dropped 'year' column (constant value)")

# Encode categorical variables
print("\n   Encoding categorical variables...")
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        print(f"      • {col}: {len(le.classes_)} categories encoded")

print("   ✓ All categorical variables encoded")

# ============================================================================
# STEP 3: EXPLORATORY DATA ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("STEP 3: EXPLORATORY DATA ANALYSIS")
print("="*70)

# Visualization 1: Correlation Heatmap
print("\n📊 Visualization 1: Feature Correlation Heatmap")

plt.figure(figsize=(14, 10))
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()

sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='RdYlGn', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Get correlations with FWI
correlations = numeric_df.corr()['FWI'].sort_values(ascending=False)
print("\nTop 10 Features Correlated with FWI:")
for i, (feat, corr) in enumerate(correlations[1:11].items(), 1):
    emoji = "🔥" if abs(corr) > 0.7 else "✓" if abs(corr) > 0.5 else "○"
    print(f"   {i:2d}. {emoji} {feat:15s}: {corr:6.4f}")

# Visualization 2: Distribution of Target Variable
print("\n📊 Visualization 2: FWI Distribution")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Original distribution
ax1.hist(df['FWI'], bins=40, color='coral', edgecolor='black', alpha=0.7)
ax1.axvline(df['FWI'].mean(), color='red', linestyle='--', linewidth=2, 
           label=f'Mean: {df["FWI"].mean():.2f}')
ax1.axvline(df['FWI'].median(), color='green', linestyle='--', linewidth=2,
           label=f'Median: {df["FWI"].median():.2f}')
ax1.set_xlabel('Fire Weather Index (FWI)', fontweight='bold', fontsize=12)
ax1.set_ylabel('Frequency', fontweight='bold', fontsize=12)
ax1.set_title('FWI Distribution', fontweight='bold', fontsize=14)
ax1.legend()
ax1.grid(alpha=0.3)

# Box plot
ax2.boxplot(df['FWI'], vert=True, patch_artist=True,
           boxprops=dict(facecolor='lightblue', alpha=0.7))
ax2.set_ylabel('Fire Weather Index (FWI)', fontweight='bold', fontsize=12)
ax2.set_title('FWI Box Plot', fontweight='bold', fontsize=14)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# STEP 4: ADVANCED FEATURE ENGINEERING
# ============================================================================
print("\n" + "="*70)
print("STEP 4: ADVANCED FEATURE ENGINEERING")
print("="*70)

df_engineered = df.copy()

# Key interaction features based on domain knowledge
print("\n   Creating interaction features...")

# Temperature-Humidity interactions (critical for fire risk)
df_engineered['Temp_RH_interaction'] = df['Temperature'] * (100 - df['RH']) / 100
df_engineered['Temp_squared'] = df['Temperature'] ** 2

# Wind-related features
df_engineered['Ws_squared'] = df['Ws'] ** 2

# Moisture code interactions
df_engineered['DMC_DC_interaction'] = df['DMC'] * df['DC'] / 100
df_engineered['FFMC_DMC'] = df['FFMC'] * df['DMC'] / 100

# Temperature-Wind interaction
df_engineered['Temp_Ws'] = df['Temperature'] * df['Ws']

# Drought index
df_engineered['Drought_index'] = (df['DC'] + df['DMC']) / 2

# Seasonal features
df_engineered['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df_engineered['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

print(f"   ✓ Created {len(df_engineered.columns) - len(df.columns)} new features")
print(f"   ✓ Total features: {len(df_engineered.columns) - 1}")

# ============================================================================
# STEP 5: PREPARE DATA FOR MODELING
# ============================================================================
print("\n" + "="*70)
print("STEP 5: DATA PREPARATION")
print("="*70)

# Separate features and target
X = df_engineered.drop('FWI', axis=1)
y = df_engineered['FWI']

print(f"\n   Features: {X.shape[1]}")
print(f"   Samples: {len(X)}")
print(f"   Target: FWI (Fire Weather Index)")

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n   Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.0f}%)")
print(f"   Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.0f}%)")

# ============================================================================
# STEP 6: TRAIN LINEAR REGRESSION MODEL
# ============================================================================
print("\n" + "="*70)
print("STEP 6: LINEAR REGRESSION TRAINING")
print("="*70)

# Train the model
print("\n   ⏳ Training Linear Regression model...")
model = LinearRegression()
model.fit(X_train, y_train)
print("   ✓ Model trained successfully!")

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate metrics
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
mae = mean_absolute_error(y_test, y_test_pred)
mape = np.mean(np.abs((y_test - y_test_pred) / (y_test + 0.01))) * 100

# Cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

# ============================================================================
# STEP 7: RESULTS & EVALUATION
# ============================================================================
print("\n" + "="*70)
print("MODEL PERFORMANCE RESULTS")
print("="*70)

print(f"\n📊 LINEAR REGRESSION PERFORMANCE:")
print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"   Training R² Score:    {train_r2:.4f} ({train_r2*100:.2f}%)")
print(f"   Test R² Score:        {test_r2:.4f} ({test_r2*100:.2f}%)")
print(f"   Cross-Val R² (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"   Test RMSE:            {test_rmse:.3f}")
print(f"   Test MAE:             {mae:.3f}")
print(f"   Test MAPE:            {mape:.2f}%")
print(f"   Overfitting Gap:      {train_r2 - test_r2:.4f}")
print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

# Success message
if test_r2 >= 0.90:
    print(f"\n✅ 🎉 SUCCESS! Achieved {test_r2*100:.2f}% R² Score!")
    print(f"   The model explains {test_r2*100:.2f}% of FWI variance!")
    print(f"   Prediction accuracy is EXCELLENT!")
elif test_r2 >= 0.85:
    print(f"\n✅ EXCELLENT! Achieved {test_r2*100:.2f}% R² Score!")
    print(f"   Very close to 90% target!")
elif test_r2 >= 0.80:
    print(f"\n✓ VERY GOOD! Achieved {test_r2*100:.2f}% R² Score!")
else:
    print(f"\n⚠️ Achieved {test_r2*100:.2f}% R² Score")

# ============================================================================
# STEP 8: FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*70)

# Get feature coefficients
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

print("\nTop 10 Most Important Features:")
for i, row in feature_importance.head(10).iterrows():
    print(f"   {i+1:2d}. {row['Feature']:20s}: {row['Coefficient']:8.4f}")

# ============================================================================
# STEP 9: VISUALIZATIONS
# ============================================================================
print("\n" + "="*70)
print("DETAILED VISUALIZATIONS")
print("="*70)

# Visualization 3: Feature Importance
print("\n📊 Visualization 3: Feature Importance (Top 15)")

plt.figure(figsize=(12, 8))
top_features = feature_importance.head(15)
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_features)))

plt.barh(range(len(top_features)), top_features['Coefficient'].abs(), 
         color=colors, edgecolor='black', alpha=0.8)
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.xlabel('Absolute Coefficient Value', fontweight='bold', fontsize=12)
plt.title('Feature Importance in Linear Regression', fontweight='bold', fontsize=14)
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

# Visualization 4: Actual vs Predicted
print("\n📊 Visualization 4: Actual vs Predicted FWI")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Scatter plot
ax1.scatter(y_test, y_test_pred, alpha=0.7, s=60, color='steelblue', 
           edgecolor='black', linewidth=0.8)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         '--', color='red', linewidth=2, label='Perfect Prediction')
ax1.set_xlabel('Actual FWI', fontweight='bold', fontsize=12)
ax1.set_ylabel('Predicted FWI', fontweight='bold', fontsize=12)
ax1.set_title('Linear Regression: Actual vs Predicted', fontweight='bold', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(alpha=0.3)

# Add R² annotation
r2_text = f'R² = {test_r2:.4f}\nAccuracy: {test_r2*100:.2f}%'
ax1.text(0.05, 0.95, r2_text, transform=ax1.transAxes, 
         fontsize=13, fontweight='bold', va='top',
         bbox=dict(boxstyle='round', facecolor='lightgreen' if test_r2 >= 0.9 else 'yellow', alpha=0.9))

# Residual plot
residuals = y_test - y_test_pred
ax2.scatter(y_test_pred, residuals, alpha=0.7, s=60, color='coral', 
           edgecolor='black', linewidth=0.8)
ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax2.set_xlabel('Predicted FWI', fontweight='bold', fontsize=12)
ax2.set_ylabel('Residuals (Actual - Predicted)', fontweight='bold', fontsize=12)
ax2.set_title('Residual Plot', fontweight='bold', fontsize=14)
ax2.grid(alpha=0.3)

# Add residual statistics
residual_text = f'Mean: {residuals.mean():.3f}\nStd: {residuals.std():.3f}'
ax2.text(0.05, 0.95, residual_text, transform=ax2.transAxes,
         fontsize=11, va='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.show()

# Visualization 5: Prediction Error Analysis
print("\n📊 Visualization 5: Prediction Error Distribution")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Error histogram
errors = y_test_pred - y_test
ax1.hist(errors, bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
ax1.axvline(errors.mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean Error: {errors.mean():.3f}')
ax1.axvline(0, color='green', linestyle='--', linewidth=2, label='Zero Error')
ax1.set_xlabel('Prediction Error', fontweight='bold', fontsize=12)
ax1.set_ylabel('Frequency', fontweight='bold', fontsize=12)
ax1.set_title('Distribution of Prediction Errors', fontweight='bold', fontsize=14)
ax1.legend()
ax1.grid(alpha=0.3)

# Percentage error histogram
percentage_errors = (errors / (y_test + 0.01)) * 100
ax2.hist(percentage_errors, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
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

# Visualization 6: Prediction vs Actual (Line Plot)
print("\n📊 Visualization 6: Prediction Comparison (First 50 Samples)")

plt.figure(figsize=(16, 6))
sample_size = min(50, len(y_test))
indices = range(sample_size)

plt.plot(indices, y_test.values[:sample_size], 'o-', label='Actual FWI', 
         color='blue', linewidth=2, markersize=6, alpha=0.7)
plt.plot(indices, y_test_pred[:sample_size], 's-', label='Predicted FWI',
         color='red', linewidth=2, markersize=6, alpha=0.7)
plt.xlabel('Sample Index', fontweight='bold', fontsize=12)
plt.ylabel('FWI Value', fontweight='bold', fontsize=12)
plt.title('Actual vs Predicted FWI (Sample Comparison)', fontweight='bold', fontsize=14)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================================
# STEP 10: EXPORT RESULTS
# ============================================================================
print("\n" + "="*70)
print("EXPORTING RESULTS")
print("="*70)

try:
    # Export 1: Predictions
    predictions_df = pd.DataFrame({
        'Actual_FWI': y_test.values,
        'Predicted_FWI': y_test_pred,
        'Error': errors.values,
        'Absolute_Error': np.abs(errors.values),
        'Percentage_Error': percentage_errors.values
    })
    predictions_df.to_csv("fwi_predictions.csv", index=False)
    print("   ✓ Saved: fwi_predictions.csv")
    
    # Export 2: Feature importance
    feature_importance.to_csv("feature_importance.csv", index=False)
    print("   ✓ Saved: feature_importance.csv")
    
    # Export 3: Model summary
    summary = {
        "Model": "Linear Regression",
        "Train_R2": train_r2,
        "Test_R2": test_r2,
        "CV_R2_Mean": cv_scores.mean(),
        "CV_R2_Std": cv_scores.std(),
        "Test_RMSE": test_rmse,
        "Test_MAE": mae,
        "Test_MAPE": mape,
        "Overfitting_Gap": train_r2 - test_r2,
        "Total_Features": X.shape[1],
        "Training_Samples": len(X_train),
        "Test_Samples": len(X_test)
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv("model_summary.csv", index=False)
    print("   ✓ Saved: model_summary.csv")
    
    print("\n✅ All results exported successfully!")
    
except Exception as e:
    print(f"   ⚠️ Export warning: {e}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("✅ ALGERIAN FOREST FIRES PREDICTION - COMPLETE!")
print("="*70)

print(f"\n🎯 FINAL RESULTS:")
print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"   Model: Linear Regression")
print(f"   Test R² Score: {test_r2:.4f} ({test_r2*100:.2f}% Accuracy)")
print(f"   Test RMSE: {test_rmse:.3f}")
print(f"   Test MAE: {mae:.3f}")
print(f"   Mean Percentage Error: {mape:.2f}%")
print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

print(f"\n📁 FILES CREATED:")
print("   • fwi_predictions.csv (Predictions with errors)")
print("   • feature_importance.csv (Feature coefficients)")
print("   • model_summary.csv (Complete metrics)")

print(f"\n📊 VISUALIZATIONS DISPLAYED:")
print("   1. Feature Correlation Heatmap")
print("   2. FWI Distribution & Box Plot")
print("   3. Feature Importance (Top 15)")
print("   4. Actual vs Predicted + Residual Plot")
print("   5. Error Distribution Analysis")
print("   6. Sample-by-Sample Comparison")

print(f"\n🎓 KEY INSIGHTS:")
print(f"   • Top predictor: {feature_importance.iloc[0]['Feature']}")
print(f"   • Model explains {test_r2*100:.2f}% of FWI variance")
print(f"   • Average prediction error: ±{mae:.3f} FWI units")
print(f"   • Strong correlation between fire indices and FWI")

if test_r2 >= 0.90:
    print(f"\n🏆 ACHIEVEMENT UNLOCKED: 90%+ R² SCORE!")
    print(f"   Your Linear Regression model is EXCELLENT!")

print("\n" + "="*70)
