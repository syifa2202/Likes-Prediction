import streamlit as st

def app():
    st.title("Tentang Saya")

    st.markdown("""
    Halo! Saya **Syifa**, seorang data enthusiast yang tertarik pada analisis media sosial, khususnya di bidang sponsorship dan engagement.  
    Saat ini saya fokus mengolah dan memodelkan data untuk memahami faktor-faktor yang memengaruhi performa konten bersponsor di Instagram, TikTok, YouTube, Bilibili dan Rednote.

    ### 💼 Pengalaman
    - Magang di DJPb & KPPN Banda Aceh
    - Mengembangkan proyek analisis data menggunakan Python & SQL
    - Membuat dashboard interaktif dengan Power BI & Tableau

    ### 🛠️ Keahlian
    - Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn)
    - SQL
    - Power BI, Tableau
    - Exploratory Data Analysis (EDA)
    - Machine Learning untuk prediksi performa konten

    ### 🔍 Topik Favorit
    - Analisis sponsorship & engagement di media sosial
    - Faktor-faktor yang memengaruhi jumlah likes
    - Visualisasi tren audiens dan interaksi
    - Prediksi performa konten menggunakan machine learning

    ---
    Aplikasi ini dibuat untuk menganalisis dan memprediksi *likes* pada postingan bersponsor
    berdasarkan berbagai fitur yang berkaitan dengan platform, tipe konten, audiens, dan strategi sponsorship.
    """)

if __name__ == "__main__":
    app()
