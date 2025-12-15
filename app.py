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
DATA_FILE = 'dataset_banjir.csv'

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
        # PERBAIKAN KRITIS 1: Hapus 'header=1' agar Pandas membaca baris pertama sebagai header (header=0 default)
        # Baris pertama berisi: Kabupaten/Kota;Banjir;Banjir Bandang;curah_hujan
        df = pd.read_csv(file_path, sep=';') 
    except FileNotFoundError:
        st.error(f"❌ File data tidak ditemukan di: {file_path}. Cek path GitHub Anda.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error saat membaca data: {e}. Cek delimiter.")
        return pd.DataFrame()

    # PERBAIKAN KRITIS 2: Hapus semua kolom yang seluruhnya kosong (berasal dari ;;;;; di CSV)
    df = df.dropna(axis=1, how='all')
    df = df.drop_duplicates()
    
    # PERBAIKAN KRITIS 3: Menstandardisasi nama kolom
    try:
        # Nama kolom dari CSV: ['Kabupaten/Kota', 'Banjir', 'Banjir Bandang', 'curah_hujan']
        # Rename agar mudah dipanggil dan menghilangkan spasi/kapitalisasi
        df.rename(columns={
            'Kabupaten/Kota': 'Kabupaten_Kota',
            'Banjir Bandang': 'Banjir_Bandang',
            'curah_hujan': 'Curah_Hujan' # Mengubah huruf kecil ke kapital
        }, inplace=True)
    except Exception:
        # Jika renaming gagal, mungkin struktur kolom berbeda, kembalikan DataFrame kosong
        return pd.DataFrame()

    kolom_numerik = ['Banjir', 'Banjir_Bandang', 'Curah_Hujan']
    for col in kolom_numerik:
        # Mengatasi desimal koma (,) menjadi titik (.)
        if df[col].dtype == 'object':
             df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
        # Konversi ke numerik, memaksa error ke NaN
        df[col] = pd.to_numeric(df[col], errors='coerce') 

    # Mengisi nilai yang hilang/error konversi dengan 0 dan menghitung total bencana
    df[kolom_numerik] = df[kolom_numerik].fillna(0)
    df['Total_Bencana'] = df['Banjir'] + df['Banjir_Bandang']

    return df

# --- 3. Tampilkan Data dan Mulai Dashboard ---
df_data = load_and_clean_data(DATA_FILE)

if not df_data.empty:
    st.sidebar.success("✅ Data berhasil dimuat dan dibersihkan.")
    
    # Tampilkan 5 baris data pertama di sidebar (opsional)
    st.sidebar.subheader("Pratinjau Data")
    st.sidebar.dataframe(df_data.head(), use_container_width=True)
    
    # Lanjutkan dengan kode visualisasi dan prediksi Anda di sini
    # Contoh:
    st.subheader("Data Analisis (15 Lokasi Awal)")
    st.dataframe(df_data)

    # ... Tambahkan kode visualisasi lainnya (misalnya plot sns atau plt)
    
else:
    st.error("Gagal memuat atau memproses data. Cek file CSV dan parameter 'sep=;'.")
