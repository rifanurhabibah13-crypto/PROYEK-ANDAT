
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Konfigurasi Awal ---
st.set_page_config(
    page_title="Dashboard Prediksi Bencana Banjir",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Judul Dashboard
st.title("🌊 Proyek UAS: Dashboard Analisis & Prediksi Bencana Banjir")
st.markdown("### Analisis Data Bencana Alam berdasarkan Curah Hujan")

# --- 2. Muat Data dan Model ---

MODEL_FILE = 'model_prediksi_banjir.pkl'
# PASTIKAN NAMA FILE DATA ANDA SUDAH BENAR DI SINI!
DATA_FILE = 'dataset_banjir (1).csv'

# Muat Model Prediksi
@st.cache_resource
def load_model(file):
    try:
        return joblib.load(file)
    except Exception as e:
        st.sidebar.error(f"❌ Error memuat model ({file}): {e}")
        return None

model = load_model(MODEL_FILE)

# Fungsi untuk memuat dan membersihkan data
@st.cache_data
def load_and_clean_data(file_path):
    try:
        df = pd.read_csv(file_path, sep=';', header=1)
    except FileNotFoundError:
        return pd.DataFrame()
    except Exception:
        df = pd.read_csv(file_path, sep=';')

    df = df.dropna(axis=1, how='all')
    df = df.drop_duplicates()

    try:
        df.columns = ['Kabupaten_Kota', 'Banjir', 'Banjir_Bandang', 'Curah_Hujan']
    except ValueError:
        return pd.DataFrame()

    kolom_numerik = ['Banjir', 'Banjir_Bandang', 'Curah_Hujan']
    for col in kolom_numerik:
        if df[col].dtype == 'object':
             df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df[kolom_numerik] = df[kolom_numerik].fillna(0)
    df['Total_Bencana'] = df['Banjir'] + df['Banjir_Bandang']

    return df

df = load_and_clean_data(DATA_FILE)

# --- 3. Bagian 1: Eksplorasi Data (EDA) ---
st.header("1. Hasil Analisis Data Eksplorasi (EDA)")

if df.empty:
     st.error("Gagal memuat data. Mohon periksa nama file data Anda di Colab.")
else:
    col1, col2 = st.columns(2)

    # Grafik 1: Top 10 Bencana
    with col1:
        st.subheader("Kabupaten/Kota Paling Terdampak")
        df_top = df.sort_values(by='Total_Bencana', ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x='Total_Bencana', y='Kabupaten_Kota', data=df_top, palette='Reds_d', ax=ax)
        ax.set_xlabel('Jumlah Kejadian')
        ax.set_ylabel('Kabupaten/Kota')
        st.pyplot(fig)

    # Grafik 2: Hubungan Curah Hujan vs Bencana
    with col2:
        st.subheader("Hubungan Curah Hujan dan Total Bencana")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.regplot(x='Curah_Hujan', y='Total_Bencana', data=df,
                    scatter_kws={'alpha':0.6}, line_kws={'color':'red'}, ax=ax)
        ax.set_xlabel('Curah Hujan (mm)')
        ax.set_ylabel('Total Bencana')
        st.pyplot(fig)

    st.subheader("Tabel Data Bersih")
    st.dataframe(df, use_container_width=True)


# --- 4. Bagian 2: Deployment Model Prediksi ---
st.header("2. Deploy Model Prediksi")

if model is not None and not df.empty:
    max_curah = df['Curah_Hujan'].max() if not df.empty else 300
    min_curah = df['Curah_Hujan'].min() if not df.empty else 0
    median_curah = df['Curah_Hujan'].median() if not df.empty else 150.0

    # Input User menggunakan Streamlit Widget (Slider)
    curah_hujan_input = st.slider(
        'Masukkan Nilai Curah Hujan (mm):',
        min_value=float(min_curah),
        max_value=float(max_curah + 50),
        value=float(median_curah),
        step=0.1,
        key='slider_hujan' # Tambahkan key untuk stabilitas
    )

    if st.button("Lakukan Prediksi"):
        # Semua baris ini DIJAMIN menggunakan 4 SPASI indentasi
        input_data = pd.DataFrame({'Curah_Hujan': [curah_hujan_input]})

        # Lakukan Prediksi
        prediksi_mentah = model.predict(input_data)[0]
        prediksi_final = max(0, prediksi_mentah)

        st.success(f"✅ Prediksi Selesai!")
        st.metric(
            label=f"Total Prediksi Kejadian Bencana (Banjir + Banjir Bandang) untuk Curah Hujan {curah_hujan_input:.2f} mm",
            value=f"{prediksi_final:.0f} Kejadian",
            delta=f"Nilai riil mungkin berupa hitungan bulat"
        )
else:
    st.warning("Model belum dimuat atau data tidak ditemukan. Prediksi tidak dapat dilakukan.")
