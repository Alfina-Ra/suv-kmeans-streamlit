import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

import matplotlib.pyplot as plt

# =========================================================
# KONFIGURASI HALAMAN STREAMLIT
# =========================================================
st.set_page_config(
    page_title="K-Means SUV Analyzer & Recommendation",
    page_icon="🚗",
    layout="wide"
)

st.markdown("""
    <style>
    .main-title { font-size: 28px; font-weight: bold; color: #1D9BF0; }
    .sub-text { font-size: 14px; color: #94A3B8; }
    </style>
""", unsafe_allow_html=True)

FEATURES = ["Hp", "Torque_Nm", "Fuel_Eco_km/l", "0_100", "Curb_W_kg"]

# =========================================================
# FUNGSI HELPER & ANALISIS
# =========================================================
@st.cache_data
def load_and_preprocess(file):
    df_raw = pd.read_csv(file)
    
    # Filter kategori SUV jika kolom Category ada
    if "Category" in df_raw.columns:
        df_suv = df_raw[df_raw["Category"].astype(str).str.upper() == "SUV"].copy()
    else:
        df_suv = df_raw.copy()
        
    df_features = df_suv[FEATURES].copy()
    for col in FEATURES:
        df_features[col] = pd.to_numeric(df_features[col], errors="coerce")
        
    df_clean = df_features.dropna().reset_index(drop=True)
    
    # Simpan juga data lengkap dengan nama mobil jika ada kolom seperti 'Model' atau 'Name'
    name_col = None
    for col in df_suv.columns:
        if col.lower() in ['model', 'name', 'car', 'suv_name', 'vehicle']:
            name_col = col
            break
            
    if name_col:
        df_clean[name_col] = df_suv[name_col].values[df_clean.index]
        
    return df_raw, df_clean, name_col

def hitung_auto_elbow(data_scaled):
    max_k = min(10, len(data_scaled) - 1)
    if max_k < 2:
        return 3, [], []

    k_values = list(range(2, max_k + 1))
    wcss_values = []

    for k in k_values:
        model = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        model.fit(data_scaled)
        wcss_values.append(model.inertia_)

    # Hitung titik elbow otomatis
    x_norm = (np.array(k_values) - min(k_values)) / (max(k_values) - min(k_values))
    y_norm = (np.array(wcss_values) - min(wcss_values)) / (max(wcss_values) - min(wcss_values))
    points = np.column_stack((x_norm, y_norm))
    
    line = points[-1] - points[0]
    distances = np.abs(np.cross(line, points[0] - points)) / np.linalg.norm(line)
    best_index = int(np.argmax(distances))
    
    return int(k_values[best_index]), k_values, wcss_values

# =========================================================
# TAMPILAN UTAMA (SIDEBAR & NAVIGATION)
# =========================================================
st.markdown('<p class="main-title">K-Means SUV Analyzer & Recommendation System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Analisis Performa & Efisiensi Mobil SUV</p>', unsafe_allow_html=True)
st.divider()

with st.sidebar:
    st.header("⚙️ Pengaturan & Data")
    uploaded_file = st.file_uploader("Upload Dataset CSV SUV", type=["csv"])
    
    st.divider()
    st.subheader("Pengaturan Klaster")
    k_mode = st.radio("Metode Penentuan K:", ["Otomatis (Elbow Method)", "Manual (Slider)"])
    
    if k_mode == "Manual (Slider)":
        manual_k = st.slider("Jumlah Klaster (K):", 2, 8, 5)
    
    st.markdown("---")
    st.info("")

# =========================================================
# LOGIKA UTAMA APLIKASI
# =========================================================
if uploaded_file is not None:
    try:
        df_raw, df_clean, name_col = load_and_preprocess(uploaded_file)
    except KeyError as e:
        st.error(
            f"Kolom wajib tidak ditemukan: {e}. "
            f"Pastikan CSV memuat kolom: {', '.join(FEATURES)}"
        )
        st.stop()

    # Preprocessing & Scaling
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_clean[FEATURES])

    # Tentukan K
    auto_k, k_vals, wcss_vals = hitung_auto_elbow(data_scaled)
    selected_k = manual_k if k_mode == "Manual (Slider)" else auto_k

    # Jalankan KMeans
    kmeans = KMeans(n_clusters=selected_k, init="k-means++", n_init=10, random_state=42)
    labels = kmeans.fit_predict(data_scaled)

    df_result = df_clean.copy()
    df_result["Cluster"] = labels

    # Metrik Evaluasi
    wcss = kmeans.inertia_
    silhouette = silhouette_score(data_scaled, labels)
    dbi = davies_bouldin_score(data_scaled, labels)
    
    # --- TAB MENU UTAMA ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Ringkasan & Metrik", 
        "📈 Visualisasi & Elbow", 
        "📋 Data & Tabel Klaster", 
        "🎯 Sistem Rekomendasi SUV"
    ])
    
    with tab1:
        st.subheader("Hasil Evaluasi Model K-Means")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Jumlah Klaster (K)", selected_k)
        col2.metric("WCSS (Inertia)", f"{wcss:.2f}")
        col3.metric("Silhouette Score", f"{silhouette:.3f}")
        col4.metric("Davies-Bouldin Index", f"{dbi:.3f}")
        
        st.divider()
        st.subheader("Profil Rata-Rata Tiap Klaster")
        
        # Hitung rata-rata per klaster
        cluster_summary = df_result.groupby("Cluster")[FEATURES].mean()
        cluster_counts = df_result["Cluster"].value_counts().sort_index()
        cluster_summary["Jumlah_Mobil"] = cluster_counts
        
        st.dataframe(cluster_summary.round(2), use_container_width=True)
        
    with tab2:
        col_v1, col_v2 = st.columns(2)
        
        with col_v1:
            st.subheader("Grafik Sebaran (Scatter Plot)")
            fig, ax = plt.subplots(figsize=(6, 4.5))
            colors = ["#1D9BF0", "#22C55E", "#F59E0B", "#A855F7", "#EF4444", "#38BDF8"]
            
            for c_id in range(selected_k):
                subset = df_result[df_result["Cluster"] == c_id]
                ax.scatter(subset["Hp"], subset["Fuel_Eco_km/l"], alpha=0.8, s=50, 
                           color=colors[c_id % len(colors)], label=f"Klaster {c_id}")
                           
            ax.set_xlabel("Tenaga Mesin (Hp)")
            ax.set_ylabel("Efisiensi BBM (km/l)")
            ax.set_title(f"Sebaran Klaster SUV (K={selected_k})")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.2)
            st.pyplot(fig)
            
        with col_v2:
            st.subheader("Grafik Elbow Method")
            fig2, ax2 = plt.subplots(figsize=(6, 4.5))
            if len(k_vals) > 0:
                ax2.plot(k_vals, wcss_vals, marker="o", color="#1D9BF0", linewidth=2)
                ax2.axvline(auto_k, color="#F59E0B", linestyle="--", label=f"Auto K = {auto_k}")
                ax2.set_xlabel("Jumlah Klaster (K)")
                ax2.set_ylabel("WCSS")
                ax2.set_title("Optimasi K dengan Elbow Method")
                ax2.legend()
                ax2.grid(True, alpha=0.2)
            st.pyplot(fig2)

    with tab3:
        st.subheader("Eksplorasi Dataset Bersih & Hasil Klaster")
        st.write(f"Total data yang diproses: **{len(df_result)} kendaraan**")
        st.dataframe(df_result, use_container_width=True, height=400)
        
        # Tombol Download CSV
        csv_data = df_result.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Hasil Klaster (CSV)",
            data=csv_data,
            file_name="hasil_klaster_suv.csv",
            mime="text/csv",
        )

    with tab4:
        st.subheader("🎯 Sistem Rekomendasi Mobil SUV Berdasarkan Preferensi")
        st.write("Masukkan kriteria kendaraan impianmu, dan sistem akan menampilkan SUV yang memenuhi kriteria, beserta klasternya!")
        
        r_col1, r_col2 = st.columns(2)
        with r_col1:
            pref_hp = st.slider("Minimal Tenaga Mesin (Hp):", int(df_clean["Hp"].min()), int(df_clean["Hp"].max()), 200)
        with r_col2:
            pref_fuel = st.slider("Minimal Efisiensi BBM (km/l):", float(df_clean["Fuel_Eco_km/l"].min()), float(df_clean["Fuel_Eco_km/l"].max()), 12.0)
            
        if st.button("🔍 Cari Rekomendasi SUV", type="primary"):
            # Filter berdasarkan preferensi pengguna
            recommendations = df_result[
                (df_result["Hp"] >= pref_hp) & 
                (df_result["Fuel_Eco_km/l"] >= pref_fuel)
            ]
            
            if len(recommendations) > 0:
                st.success(f"Ditemukan {len(recommendations)} kendaraan yang sesuai dengan kriteria kamu!")
                if name_col:
                    st.dataframe(recommendations[[name_col, "Cluster", "Hp", "Torque_Nm", "Fuel_Eco_km/l", "0_100"]], use_container_width=True)
                else:
                    st.dataframe(recommendations, use_container_width=True)
            else:
                st.warning("⚠️ Tidak ditemukan SUV yang mencocokkan kriteria tersebut secara persis. Coba longgarkan rentang filter tenaga atau efisiensi BBM.")

else:
    st.info("👋 Silakan **Upload Dataset CSV SUV** melalui panel di sebelah kiri (Sidebar) untuk memulai analisis dan visualisasi.")
    
    with st.expander("ℹ️ Panduan Format Kolom Dataset"):
        st.write("Pastikan file CSV memiliki kolom-kolom numerik berikut:")
        st.code(", ".join(FEATURES))
        st.write("Opsional: Kolom nama/model kendaraan (misal: `Model` atau `Name`) untuk fitur rekomendasi.")