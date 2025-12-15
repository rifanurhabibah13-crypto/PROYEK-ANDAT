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
# UBAH NAMA FILE INI AGAR SESUAI DENGAN YANG ANDA UPLOAD KE GITHUB
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
        # PERBAIKAN 1: Hapus header=1
        df = pd.read_csv(file_path, sep=';') 
    except FileNotFoundError:
        st.error(f"❌ Error: File data tidak ditemukan di: {file_path}. Cek nama file dan path di GitHub.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error saat membaca data: {e}. Cek delimiter (';').")
        return pd.DataFrame()

    # PERBAIKAN 2: Membersihkan dan menstandardisasi kolom
    df = df.dropna(axis=1, how='all') # Drop kolom yang seluruhnya kosong (penting!)
    df = df.drop_duplicates()

    # Standardisasi nama kolom dari header CSV yang terbaca
    try:
        df.rename(columns={
            'Kabupaten/Kota': 'Kabupaten_Kota',
            'Banjir Bandang': 'Banjir_Bandang',
            'curah_hujan': 'Curah_Hujan' # Mengubah kapitalisasi
        }, inplace=True)
    except Exception:
        # Lanjutkan jika renaming gagal, mungkin data sudah dimuat tapi kolomnya kacau
        pass

    # Hanya ambil 4 kolom utama yang kita butuhkan
    COLUMNS_TO_KEEP = ['Kabupaten_Kota', 'Banjir', 'Banjir_Bandang', 'Curah_Hujan']
    # Filter hanya kolom yang ada di DataFrame setelah renaming
    df = df[[col for col in COLUMNS_TO_KEEP if col in df.columns]]
    
    kolom_numerik = ['Banjir', 'Banjir_Bandang', 'Curah_Hujan']
    for col in kolom_numerik:
        if col in df.columns:
            # Mengatasi koma (,) sebagai desimal
            if df[col].dtype == 'object':
                 df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df[kolom_numerik] = df[kolom_numerik].fillna(0)
    
    # Hapus baris yang mungkin kosong/rusak akibat data di tengah yang kacau
    df = df.dropna(subset=['Kabupaten_Kota'])

    df['Total_Bencana'] = df['Banjir'] + df['Banjir_Bandang']

    return df

# --- 3. Tampilkan Data dan Mulai Dashboard ---
df_data = load_and_clean_data(DATA_FILE)

# ... Lanjutkan sisa kode Streamlit Anda untuk menampilkan dashboard ...
if not df_data.empty:
    st.sidebar.success("✅ Data berhasil dimuat dan dibersihkan.")
    # Tampilkan 5 baris data pertama di sidebar (opsional)
    st.sidebar.subheader("Pratinjau Data")
    st.sidebar.dataframe(df_data.head(), use_container_width=True)

    # Tambahkan visualisasi jika ada
    st.subheader("Data Analisis")
    st.dataframe(df_data, use_container_width=True)

    # ... Tambahkan kode visualisasi dan prediksi Anda ...
    
    if model is not None:
        # Logika Prediksi Anda di sini
        st.header("3. Prediksi Bencana")
        # Kode prediksi Anda di sini (Pastikan identasi 4 SPASI)
        
        # ...
        
    else:
        st.warning("Model Prediksi belum berhasil dimuat.")

else:
    st.error("Gagal memuat atau memproses data. Cek file CSV dan parameter 'sep=;'.")
