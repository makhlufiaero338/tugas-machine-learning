# 1. Mengimpor pustaka yang diperlukan
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Memuat dataset
# Asumsikan dataset adalah CreditDataset.csv yang memiliki kolom fitur dan target klasifikasi (misalkan: 'default' untuk prediksi)
data = pd.read_csv('CreditDataset.csv')

# 3. Menampilkan beberapa baris dari dataset untuk memahami strukturnya
print(data.head())

# 4. Preprocessing data
# Asumsikan ada kolom kategorikal yang perlu diubah menjadi numerik, dan data tidak memiliki nilai kosong

# Memeriksa jika ada nilai kosong
print(data.isnull().sum())

# Jika ada kolom kategorikal, gunakan pd.get_dummies untuk mengubahnya ke numerik
data = pd.get_dummies(data, drop_first=True)

# Memisahkan fitur (X) dan target (y)
# Misalkan kolom terakhir adalah target (y) dan lainnya adalah fitur (X)
X = data.iloc[:, :-1].values  # Semua kolom kecuali yang terakhir
y = data.iloc[:, -1].values   # Kolom terakhir sebagai target

# 5. Membagi dataset menjadi data latih (70%) dan data uji (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 6. Standarisasi fitur (optional, tapi umumnya penting untuk model seperti Logistic Regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7. Melatih model klasifikasi (menggunakan Logistic Regression dalam hal ini)
model = LogisticRegression()
model.fit(X_train, y_train)

# 8. Prediksi pada data uji
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilitas untuk kelas positif (1)

# 9. Mengevaluasi model menggunakan berbagai metrik evaluasi

# Akurasi
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi: {accuracy}")

# Presisi
precision = precision_score(y_test, y_pred)
print(f"Presisi: {precision}")

# Recall
recall = recall_score(y_test, y_pred)
print(f"Recall: {recall}")

# F1-Score
f1 = f1_score(y_test, y_pred)
print(f"F1-Score: {f1}")

# AUC (Area Under Curve)
auc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC: {auc}")

# 10. Menggambar kurva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# 11. Visualisasi Confusion Matrix untuk melihat performa prediksi
from sklearn.metrics import confusion_matrix

# Membuat confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.show()

# 12. Menyimpan hasil ke file .ipynb (optional, untuk disimpan dan diunggah ke GitHub)
import nbformat as nbf

# Menyimpan file ini sebagai .ipynb
nb = nbf.v4.new_notebook()
cells = [
    nbf.v4.new_code_cell("""\
# Kode Python Klasifikasi dan Evaluasi Model dengan Scikit-learn
# ... (Paste kode di sini)""")
]

nb['cells'] = cells
with open("classification_model.ipynb", 'w') as f:
    nbf.write(nb, f)

print("Notebook saved as classification_model.ipynb")
\
