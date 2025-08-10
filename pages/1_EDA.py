import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def app():
    st.title("📈 Exploratory Data Analysis (EDA)")
    
    # Load dataset
    df = pd.read_csv("data/dataset_original.csv")

    # Filter konten bersponsor dan non-sponsor
    df_sponsored = df[df["is_sponsored"] == True]
    df_non_sponsored = df[df["is_sponsored"] == False]

    st.subheader("1. Perbandingan Engagement: Konten Bersponsor vs Non-Sponsor")
    engagement_metrics = ["views", "likes", "shares", "comments_count"]

    for metric in engagement_metrics:
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x="is_sponsored", y=metric, ax=ax)
        ax.set_title(f"Distribusi {metric.title()} Berdasarkan Sponsorship")
        ax.set_xticklabels(["Non-Sponsor", "Sponsor"])
        st.pyplot(fig)

    st.subheader("2. Engagement Berdasarkan Disclosure Type")
    metric = st.selectbox("Pilih metrik:", ["likes", "views"])
    fig, ax = plt.subplots()
    sns.boxplot(data=df_sponsored, x="disclosure_type", y=metric, ax=ax)
    ax.set_title(f"Distribusi {metric.title()} per Disclosure Type (Konten Bersponsor)")
    st.pyplot(fig)

    st.subheader("3. Efektivitas Kategori Konten Bersponsor")
    fig, ax = plt.subplots()
    sns.barplot(data=df_sponsored, x="content_category", y="likes", estimator='mean', ax=ax)
    ax.set_title("Rata-rata Likes per Kategori Konten (Bersponsor)")
    st.pyplot(fig)

    st.subheader("4. Platform dengan Engagement Tertinggi")
    fig, ax = plt.subplots()
    sns.barplot(data=df_sponsored, x="platform", y="likes", estimator='mean', ax=ax)
    ax.set_title("Rata-rata Likes per Platform (Bersponsor)")
    st.pyplot(fig)

    st.subheader("5. Tipe Konten dan Engagement")
    fig, ax = plt.subplots()
    sns.boxplot(data=df_sponsored, x="content_type", y="likes", ax=ax)
    ax.set_title("Distribusi Likes Berdasarkan Tipe Konten (Bersponsor)")
    st.pyplot(fig)

    st.subheader("6. Engagement Berdasarkan Usia Audiens")
    fig, ax = plt.subplots()
    sns.barplot(data=df_sponsored, x="audience_age_distribution", y="likes", estimator='mean', ax=ax)
    ax.set_title("Rata-rata Likes Berdasarkan Usia Audiens")
    st.pyplot(fig)

    st.subheader("7. Engagement Berdasarkan Gender Audiens")
    fig, ax = plt.subplots()
    sns.boxplot(data=df_sponsored, x="audience_gender_distribution", y="likes", ax=ax)
    ax.set_title("Distribusi Likes Berdasarkan Gender Audiens")
    st.pyplot(fig)

    st.subheader("8. Pengaruh Lokasi Audiens")
    fig, ax = plt.subplots()
    sns.barplot(data=df_sponsored, x="audience_location", y="likes", estimator='mean', ax=ax)
    ax.set_title("Rata-rata Likes Berdasarkan Lokasi Audiens")
    st.pyplot(fig)

    st.subheader("Kesimpulan Analisis Sementara")
    st.markdown("""
    Berdasarkan hasil eksplorasi data yang telah dilakukan:

    - **Tidak terdapat perbedaan signifikan** pada metrik engagement antara konten bersponsor dan non-sponsor.
    - Engagement juga relatif **serupa di berbagai platform**, disclosure type, kategori konten, dan tipe konten.
    - Variasi likes berdasarkan usia, gender, dan lokasi audiens terlihat **minim**, yang mengindikasikan faktor-faktor ini
    mungkin bukan penentu utama perbedaan engagement dalam dataset ini.
  
    **Catatan:** Temuan ini bersifat deskriptif dan perlu diuji lebih lanjut menggunakan analisis statistik atau pemodelan prediktif.
    """)

if __name__ == "__main__":
    app()
