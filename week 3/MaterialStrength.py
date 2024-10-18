# 1. Import pustaka yang diperlukan
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# 2. Memuat dataset
data = pd.read_csv('MaterialStrength.csv')

# 3. Menampilkan beberapa baris dari dataset untuk melihat strukturnya
print(data.head())

# 4. Memisahkan fitur (X) dan target (y)
# Misalkan kolom terakhir adalah targetnya (y), dan yang lainnya adalah fitur (X)
X = data.iloc[:, :-1].values  # Semua kolom kecuali yang terakhir
y = data.iloc[:, -1].values   # Kolom terakhir sebagai target

# 5. Membagi dataset menjadi data latih (70%) dan data uji (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 6. Melatih model regresi linier biasa
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# 7. Prediksi pada data uji menggunakan regresi linier biasa
y_pred_lin = lin_reg.predict(X_test)

# 8. Evaluasi model regresi linier biasa
mse_lin = mean_squared_error(y_test, y_pred_lin)
rmse_lin = np.sqrt(mse_lin)
r2_lin = r2_score(y_test, y_pred_lin)

print("\n=== Evaluasi Regresi Linier Biasa ===")
print(f"MSE: {mse_lin}")
print(f"RMSE: {rmse_lin}")
print(f"R^2: {r2_lin}")

# 9. Menggunakan Basis Fungsi: PolynomialFeatures untuk menambah kompleksitas model
poly_features = PolynomialFeatures(degree=2)  # Derajat polinomial 2
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

# 10. Melatih model regresi linier dengan basis fungsi (polinomial)
lin_reg_poly = LinearRegression()
lin_reg_poly.fit(X_train_poly, y_train)

# 11. Prediksi pada data uji menggunakan regresi linier polinomial
y_pred_poly = lin_reg_poly.predict(X_test_poly)

# 12. Evaluasi model regresi linier dengan basis fungsi (polinomial)
mse_poly = mean_squared_error(y_test, y_pred_poly)
rmse_poly = np.sqrt(mse_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print("\n=== Evaluasi Regresi Linier dengan Basis Fungsi (Polinomial) ===")
print(f"MSE: {mse_poly}")
print(f"RMSE: {rmse_poly}")
print(f"R^2: {r2_poly}")

# 13. Visualisasi Hasil
# Membandingkan hasil prediksi linier dan polinomial dengan nilai sebenarnya
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='True Values')
plt.scatter(range(len(y_test)), y_pred_lin, color='red', label='Linear Regression Predictions')
plt.scatter(range(len(y_test)), y_pred_poly, color='green', label='Polynomial Regression Predictions')
plt.title('Comparison of True vs Predicted Values (Linear vs Polynomial)')
plt.legend()
plt.show()
