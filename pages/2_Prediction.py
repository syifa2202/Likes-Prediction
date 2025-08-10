import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def mean_absolute_percentage_error(y_true, y_pred):
    """Custom fungsi MAPE"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def app():
    st.title("📊 Prediksi Likes Konten Bersponsor")
    st.write("Gunakan model Lasso terbaik untuk memprediksi jumlah likes.")

    # Load model
    model = joblib.load("best_model_lasso.pkl")

    # Load data processed
    train_df = pd.read_csv("train_processed.csv")
    test_df = pd.read_csv("test_processed.csv")

    # Pisahkan fitur & target
    target_col = "likes"
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    # 🔍 Cek & imputasi missing values di X_test (tanpa warning di UI)
    if X_test.isna().sum().sum() > 0:
        X_test = X_test.fillna(X_test.median())

    # 🔍 Cek & bersihkan missing values di y_test (tanpa warning di UI)
    if y_test.isna().sum() > 0:
        mask = ~y_test.isna()
        X_test = X_test.loc[mask]
        y_test = y_test.loc[mask]

    # Evaluasi model
    st.subheader("🔍 Evaluasi Model pada Test Set")
    y_pred = model.predict(X_test)
    st.write(f"**MAE**: {mean_absolute_error(y_test, y_pred):,.2f}")
    st.write(f"**MSE**: {mean_squared_error(y_test, y_pred):,.2f}")
    st.write(f"**RMSE**: {np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}")
    st.write(f"**MAPE**: {mean_absolute_percentage_error(y_test, y_pred):.2f}%")
    st.write(f"**R²**: {r2_score(y_test, y_pred):.4f}")

    # Prediksi baru
    st.subheader("🎯 Prediksi Baru")
    st.write("Masukkan nilai fitur untuk memprediksi likes:")

    input_data = {}
    for col in X_train.columns:
        val = st.number_input(f"{col}", value=float(X_train[col].mean()))
        input_data[col] = val

    if st.button("Prediksi Likes"):
        input_df = pd.DataFrame([input_data])

        # Imputasi NaN di input user
        if input_df.isna().sum().sum() > 0:
            input_df = input_df.fillna(X_train.median())

        prediction = model.predict(input_df)[0]
        st.success(f"Prediksi Likes: {prediction:,.0f}")

    st.subheader("📊 Insight & Rekomendasi Model")
    st.markdown("""
    **Insight:**
    - Rata-rata prediksi meleset sekitar **31 likes**.
    - Model belum mampu menjelaskan variansi data secara signifikan (R² ≈ 0).
    - Error persentase terlihat rendah karena skala likes besar.

    **Rekomendasi:**
    1. Lakukan **feature engineering** untuk membuat variabel baru yang lebih informatif.
    2. Pastikan data bersih dari missing values dan outlier.
    3. Analisis ulang fitur penting untuk meningkatkan relevansi input model.
    """)

if __name__ == "__main__":
    app()
