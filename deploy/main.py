import streamlit as st
import pandas as pd
import io
import numpy as np
from sklearn.preprocessing import StandardScaler
# Impor library untuk modeling
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from yellowbrick.cluster import KElbowVisualizer
import plotly.express as px
import matplotlib.pyplot as plt


# Konfigurasi halaman
st.set_page_config(layout="wide")

# --- STATE MANAGEMENT ---
if 'dataframes' not in st.session_state:
    st.session_state.dataframes = {} 
    st.session_state.originals = {}
    st.session_state.active_df_name = None
    st.session_state.cleaned_preview = None
    st.session_state.sidebar_mode = "Data Wrangling"
    st.session_state.modeling_results = {} 

def delete_dataframe(df_name):
    """Fungsi untuk menghapus dataframe dari semua state."""
    if df_name in st.session_state.dataframes:
        del st.session_state.dataframes[df_name]
    if df_name in st.session_state.originals:
        del st.session_state.originals[df_name]
    if st.session_state.active_df_name == df_name:
        st.session_state.active_df_name = None
        st.session_state.modeling_results = {} # Reset hasil model jika df dihapus

st.sidebar.header("⚙️ Kontrol & Manajemen Data")

# --- PERUBAHAN DI SINI: MENAMBAHKAN MODE MODELING ---
st.session_state.sidebar_mode = st.sidebar.radio(
    "Pilih Mode:",
    ["Data Wrangling", "Data Merge", "Analisis Segmentasi", "Modeling"],
    key='mode_selector'
)

if st.session_state.sidebar_mode == "Data Wrangling":
    uploaded_files = st.sidebar.file_uploader(
        "1. Unggah file CSV", type=["csv"], accept_multiple_files=True, key='uploader_wrangling'
    )
    if uploaded_files:
        for file in uploaded_files:
            if file.name not in st.session_state.dataframes:
                try:
                    df = pd.read_csv(file)
                    st.session_state.dataframes[file.name] = df.copy()
                    st.session_state.originals[file.name] = df.copy()
                    st.sidebar.success(f"File '{file.name}' dimuat.")
                except Exception as e:
                    st.sidebar.error(f"Gagal memuat {file.name}: {e}")
    if st.session_state.dataframes:
        st.sidebar.markdown("---")
        available_dfs = list(st.session_state.dataframes.keys())
        try:
            current_index = available_dfs.index(st.session_state.active_df_name)
        except (ValueError, TypeError):
            current_index = 0
            if available_dfs:
                st.session_state.active_df_name = available_dfs[0]
        st.session_state.active_df_name = st.sidebar.selectbox("Pilih DataFrame Aktif:", options=available_dfs, index=current_index, key='active_df_selector')
        st.sidebar.markdown("---")
        st.sidebar.subheader("Manajemen DataFrame")
        for df_name in list(st.session_state.dataframes.keys()):
            col1, col2 = st.sidebar.columns([0.8, 0.2])
            col1.write(f"📄 `{df_name}`")
            if col2.button("❌", key=f"del_{df_name}", help=f"Hapus DataFrame '{df_name}'"):
                delete_dataframe(df_name)
                st.rerun()
        st.sidebar.markdown("---")
        st.sidebar.subheader("Aksi")
        if st.sidebar.button("Buat Salinan DataFrame Aktif"):
            if st.session_state.active_df_name:
                copy_name = f"copy_of_{st.session_state.active_df_name}"
                i = 1
                while copy_name in st.session_state.dataframes:
                    copy_name = f"copy_of_{st.session_state.active_df_name}_{i}"
                    i += 1
                df_to_copy = st.session_state.dataframes[st.session_state.active_df_name].copy()
                st.session_state.dataframes[copy_name] = df_to_copy
                st.sidebar.success(f"Salinan '{copy_name}' dibuat.")
                st.rerun()
        if st.sidebar.button("↩️ Kembalikan ke Data Awal"):
            active_name = st.session_state.active_df_name
            if active_name in st.session_state.originals:
                st.session_state.dataframes[active_name] = st.session_state.originals[active_name].copy()
                st.session_state.cleaned_preview = None
                st.sidebar.success(f"Data '{active_name}' telah dikembalikan.")
                st.rerun()
            else:
                st.sidebar.warning(f"Tidak ada data asli untuk '{active_name}'.")

elif st.session_state.sidebar_mode in ["Data Merge", "Analisis Segmentasi", "Modeling"]:
    st.sidebar.info(f"Gunakan panel utama untuk mode '{st.session_state.sidebar_mode}'.")
    if st.session_state.dataframes:
        st.sidebar.markdown("---")
        st.sidebar.subheader("DataFrame Tersedia:")
        for df_name in st.session_state.dataframes.keys():
            st.sidebar.caption(f"📄 `{df_name}`")

st.title("🛠️ Alat Analisis & Preparasi Data Interaktif")

if st.session_state.sidebar_mode == "Data Wrangling":
    if st.session_state.active_df_name and st.session_state.active_df_name in st.session_state.dataframes:
        df_active = st.session_state.dataframes[st.session_state.active_df_name]
        st.subheader(f"Preview Data Aktif: `{st.session_state.active_df_name}`")
        st.dataframe(df_active)
        tab1, tab2, tab3 = st.tabs(["🧹 Informasi & Pembersihan", "🔬 Feature Engineering", "💾 Unduh Data"])
        with tab1:
            st.header("Informasi & Pembersihan Data")
            with st.expander("Lihat Informasi Detail Data"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Bentuk Data (Shape):**"); st.text(f"Baris: {df_active.shape[0]}, Kolom: {df_active.shape[1]}")
                    st.write("**Tipe Data Kolom (dtypes):**"); st.dataframe(df_active.dtypes.reset_index().rename(columns={'index': 'Kolom', 0: 'Tipe Data'}))
                with col2:
                    st.write("**Informasi Memori & Null (info()):**")
                    buffer = io.StringIO(); df_active.info(buf=buffer); st.text(buffer.getvalue())
            with st.expander("Lakukan Pembersihan Data (Timpa Data Aktif)", expanded=True):
                st.warning(f"Aksi di bawah ini akan mengubah DataFrame `{st.session_state.active_df_name}` secara permanen.")
                remove_duplicates = st.checkbox("Hapus baris duplikat")
                remove_nulls = st.checkbox("Hapus baris yang mengandung nilai kosong (NaN)")
                if st.button("Terapkan Pembersihan"):
                    df_cleaned = df_active.copy()
                    rows_before_total = len(df_cleaned)
                    if remove_duplicates: df_cleaned.drop_duplicates(inplace=True)
                    if remove_nulls: df_cleaned.dropna(inplace=True)
                    rows_after_total = len(df_cleaned)
                    st.session_state.cleaned_preview = df_cleaned.copy()
                    st.session_state.dataframes[st.session_state.active_df_name] = df_cleaned
                    st.success(f"Pembersihan selesai! **{rows_before_total - rows_after_total}** baris dihapus.")
                    st.rerun()
            if st.session_state.cleaned_preview is not None:
                with st.expander("Hasil Setelah Pembersihan", expanded=True):
                    st.dataframe(st.session_state.cleaned_preview)
                    st.info(f"Bentuk data setelah dibersihkan: **{st.session_state.cleaned_preview.shape[0]}** baris, **{st.session_state.cleaned_preview.shape[1]}** kolom.")
                    if st.button("Tutup Preview Hasil"):
                        st.session_state.cleaned_preview = None
                        st.rerun()
        with tab2:
            st.header("Rekayasa Fitur (Feature Engineering)")
            st.warning(f"**Perhatian:** Sebagian besar aksi di bawah ini akan **menimpa (overwrite)** DataFrame aktif (`{st.session_state.active_df_name}`).")
            with st.expander("1. Buat DataFrame Baru dengan Kondisi (Filter) atau Salin"):
                st.info("Isi kueri untuk memfilter data, atau biarkan kosong untuk membuat salinan DataFrame aktif.")
                query_string = st.text_area("Kueri Filter (Opsional):", help="Contoh: age > 30 atau city == 'Jakarta'")
                new_df_name = st.text_input("Nama DataFrame Baru:", f"new_{st.session_state.active_df_name}")
                if st.button("Buat DataFrame Baru"):
                    if not new_df_name: st.warning("Nama DataFrame baru tidak boleh kosong.")
                    elif new_df_name in st.session_state.dataframes: st.warning(f"Nama '{new_df_name}' sudah digunakan.")
                    else:
                        if query_string:
                            try:
                                new_df = df_active.query(query_string)
                                st.session_state.dataframes[new_df_name] = new_df
                                st.success(f"DataFrame baru '{new_df_name}' berhasil dibuat!"); st.rerun()
                            except Exception as e: st.error(f"Terjadi kesalahan pada kueri: {e}")
                        else:
                            new_df = df_active.copy()
                            st.session_state.dataframes[new_df_name] = new_df
                            st.success(f"Salinan '{new_df_name}' berhasil dibuat."); st.rerun()
            with st.expander("2. Buat Kolom Baru dari Operasi Matematika", expanded=True):
                st.info("Pilih dua kolom numerik dan operasi untuk membuat fitur baru.")
                numeric_cols = df_active.select_dtypes(include=np.number).columns.tolist()
                if len(numeric_cols) < 2: st.warning("Tidak cukup kolom numerik (diperlukan min 2).")
                else:
                    col1, col2, col3 = st.columns(3)
                    with col1: col_a = st.selectbox("Pilih Kolom Pertama:", options=numeric_cols, key="col_a")
                    with col2: operator = st.selectbox("Pilih Operasi:", options=['+', '-', '*', '/'], key="op")
                    with col3: col_b = st.selectbox("Pilih Kolom Kedua:", options=numeric_cols, key="col_b")
                    new_feature_name = st.text_input("Nama Kolom Baru:", f"{col_a}{operator}{col_b}")
                    if st.button("Buat Kolom Baru (Timpa)"):
                        if not new_feature_name: st.warning("Nama kolom baru tidak boleh kosong.")
                        elif new_feature_name in df_active.columns: st.warning(f"Kolom '{new_feature_name}' sudah ada.")
                        else:
                            try:
                                df_temp = df_active.copy()
                                if operator == '+': df_temp[new_feature_name] = pd.to_numeric(df_temp[col_a]) + pd.to_numeric(df_temp[col_b])
                                elif operator == '-': df_temp[new_feature_name] = pd.to_numeric(df_temp[col_a]) - pd.to_numeric(df_temp[col_b])
                                elif operator == '*': df_temp[new_feature_name] = pd.to_numeric(df_temp[col_a]) * pd.to_numeric(df_temp[col_b])
                                elif operator == '/': df_temp[new_feature_name] = pd.to_numeric(df_temp[col_a]) / pd.to_numeric(df_temp[col_b]).replace(0, np.nan)
                                st.session_state.dataframes[st.session_state.active_df_name] = df_temp
                                st.success(f"Kolom baru '{new_feature_name}' berhasil dibuat."); st.rerun()
                            except Exception as e: st.error(f"Gagal membuat kolom baru: {e}")
            with st.expander("3. Pilih atau Hapus Kolom"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Pilih Kolom**"); cols_to_keep = st.multiselect("Pilih kolom untuk disimpan:", options=df_active.columns.tolist(), key="cols_keep")
                    if st.button("Simpan Hanya Kolom Terpilih (Timpa)"):
                        st.session_state.dataframes[st.session_state.active_df_name] = df_active[cols_to_keep]
                        st.success("DataFrame aktif diperbarui."); st.rerun()
                with col2:
                    st.write("**Hapus Kolom**"); cols_to_drop = st.multiselect("Pilih kolom untuk dihapus:", options=df_active.columns.tolist(), key="cols_drop")
                    if st.button("Hapus Kolom Terpilih (Timpa)"):
                        st.session_state.dataframes[st.session_state.active_df_name] = df_active.drop(columns=cols_to_drop)
                        st.success("Kolom dihapus dari DataFrame aktif."); st.rerun()
            with st.expander("4. Hapus Baris Berdasarkan Kondisi"):
                col_to_filter = st.selectbox("Pilih kolom acuan:", options=df_active.columns.tolist(), key="col_filter")
                text_to_find = st.text_input("Hapus baris jika kolom acuan mengandung teks ini:")
                if st.button("Hapus Baris (Timpa)"):
                    if text_to_find:
                        df_filtered = df_active[~df_active[col_to_filter].astype(str).str.contains(text_to_find, na=False)]
                        st.session_state.dataframes[st.session_state.active_df_name] = df_filtered
                        st.success(f"Baris yang mengandung '{text_to_find}' telah dihapus."); st.rerun()
            with st.expander("5. Ubah Tipe Data ke Datetime"):
                cols_to_datetime = st.multiselect("Pilih kolom untuk diubah ke format Datetime:", options=df_active.columns.tolist(), key="cols_datetime")
                if st.button("Ubah ke Datetime (Timpa)"):
                    df_temp = df_active.copy()
                    for col in cols_to_datetime:
                        df_temp[col] = pd.to_datetime(df_temp[col], errors='coerce')
                    st.session_state.dataframes[st.session_state.active_df_name] = df_temp
                    st.success(f"Kolom terpilih telah diubah ke Datetime."); st.rerun()
            with st.expander("6. Ganti Nilai dalam Suatu Kolom"):
                col_to_replace = st.selectbox("Pilih kolom:", options=df_active.columns.tolist(), key="col_replace")
                val_to_find = st.text_input("Nilai yang ingin diganti:")
                val_to_replace = st.text_input("Nilai pengganti:")
                if st.button("Ganti Nilai (Timpa)"):
                    if val_to_find:
                        df_temp = df_active.copy()
                        df_temp[col_to_replace] = df_temp[col_to_replace].astype(str).replace(val_to_find, val_to_replace)
                        st.session_state.dataframes[st.session_state.active_df_name] = df_temp
                        st.success(f"Nilai telah diganti."); st.rerun()
        with tab3:
            st.header("Unduh Hasil Data")
            st.write(f"Unduh DataFrame `{st.session_state.active_df_name}` yang sedang aktif.")
            csv = df_active.to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 Unduh sebagai CSV", data=csv, file_name=f"{st.session_state.active_df_name}.csv", mime="text/csv")
    else:
        st.info("👈 Silakan pilih DataFrame di mode 'Data Wrangling' untuk memulai.")

elif st.session_state.sidebar_mode == "Data Merge":
    st.header("🔄 Gabungkan DataFrame (Merge)")
    available_dfs = list(st.session_state.dataframes.keys())
    if len(available_dfs) < 2:
        st.warning("Dibutuhkan minimal 2 DataFrame untuk digabungkan.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            df_left_name = st.selectbox("Pilih DataFrame Kiri:", options=available_dfs, key="df_left")
            df_left = st.session_state.dataframes[df_left_name]
            st.write("Kolom di DataFrame Kiri:"); st.dataframe(df_left.head(3))
        with col2:
            df_right_name = st.selectbox("Pilih DataFrame Kanan:", options=available_dfs, key="df_right")
            df_right = st.session_state.dataframes[df_right_name]
            st.write("Kolom di DataFrame Kanan:"); st.dataframe(df_right.head(3))
        st.markdown("---"); st.subheader("Konfigurasi Penggabungan")
        common_cols = list(set(df_left.columns) & set(df_right.columns))
        merge_key = st.multiselect("Pilih kolom kunci untuk menggabungkan:", options=common_cols)
        how_to_merge = st.selectbox("Pilih jenis penggabungan (how):", options=['inner', 'left', 'right', 'outer'])
        new_merged_df_name = st.text_input("Beri nama untuk DataFrame hasil gabungan:", f"merged_{df_left_name}_{df_right_name}")
        if st.button("Gabungkan DataFrame"):
            if not merge_key: st.warning("Mohon pilih minimal satu kolom kunci.")
            elif not new_merged_df_name: st.warning("Mohon beri nama untuk DataFrame yang baru.")
            elif new_merged_df_name in st.session_state.dataframes: st.warning(f"Nama '{new_merged_df_name}' sudah digunakan.")
            else:
                try:
                    merged_df = pd.merge(df_left, df_right, on=merge_key, how=how_to_merge)
                    st.session_state.dataframes[new_merged_df_name] = merged_df
                    st.success(f"DataFrame baru '{new_merged_df_name}' berhasil dibuat!")
                    st.write("Preview Hasil Gabungan:"); st.dataframe(merged_df.head())
                except Exception as e: st.error(f"Gagal menggabungkan: {e}")

elif st.session_state.sidebar_mode == "Analisis Segmentasi":
    st.header("💡 Analisis Segmentasi Pelanggan")
    available_dfs = list(st.session_state.dataframes.keys())
    if not available_dfs:
        st.warning("Tidak ada DataFrame yang tersedia.")
    else:
        source_df_name = st.selectbox("Pilih DataFrame sumber untuk analisis:", options=available_dfs)
        df_source = st.session_state.dataframes[source_df_name]
        required_cols = ['customer_id', 'order_id', 'order_item_id', 'price', 'freight_value', 'product_category_name', 'order_purchase_timestamp']
        missing_cols = [col for col in required_cols if col not in df_source.columns]
        if missing_cols:
            st.error(f"DataFrame '{source_df_name}' tidak memiliki kolom yang dibutuhkan: **{', '.join(missing_cols)}**")
        else:
            st.success(f"DataFrame '{source_df_name}' valid dan siap untuk dianalisis.")
            st.markdown("---")
            new_final_df_name = st.text_input("Beri nama untuk DataFrame hasil akhir (setelah standarisasi):", f"standardized_data_from_{source_df_name}")
            if st.button("Lakukan Agregasi & Standarisasi Fitur"):
                if not new_final_df_name: st.warning("Mohon beri nama untuk DataFrame hasil akhir.")
                elif new_final_df_name in st.session_state.dataframes: st.warning(f"Nama '{new_final_df_name}' sudah digunakan.")
                else:
                    try:
                        with st.spinner("Langkah 1/2: Melakukan agregasi data..."):
                            df = df_source.copy()
                            df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
                            agg_data = df.groupby('customer_id').agg(
                                num_orders=('order_id', pd.Series.nunique),
                                num_items=('order_item_id', 'sum'),
                                total_spent=('price', 'sum'),
                                avg_spent_per_order=('price', lambda x: x.sum() / df.loc[x.index, 'order_id'].nunique()),
                                avg_freight_value=('freight_value', 'mean'),
                                num_categories=('product_category_name', pd.Series.nunique),
                                first_purchase=('order_purchase_timestamp', 'min'),
                                last_purchase=('order_purchase_timestamp', 'max')
                            ).reset_index()
                            agg_data['purchase_span_days'] = (agg_data['last_purchase'] - agg_data['first_purchase']).dt.days
                            agg_data['avg_order_interval'] = agg_data['purchase_span_days'] / (agg_data['num_orders'] - 1)
                            agg_data['avg_order_interval'] = agg_data['avg_order_interval'].replace([np.inf, -np.inf], 0).fillna(0)
                        st.success("✔ Agregasi selesai.")
                        with st.expander("Lihat Hasil Agregasi (Sebelum Standarisasi)"):
                            st.dataframe(agg_data)
                        with st.spinner("Langkah 2/2: Melakukan standarisasi fitur..."):
                            features_to_scale = [
                                'num_orders', 'num_items', 'total_spent', 'avg_spent_per_order',
                                'avg_freight_value', 'num_categories', 'purchase_span_days', 'avg_order_interval'
                            ]
                            existing_features_to_scale = [f for f in features_to_scale if f in agg_data.columns]
                            data = agg_data.copy()
                            scaler = StandardScaler()
                            data[existing_features_to_scale] = scaler.fit_transform(data[existing_features_to_scale])
                        st.session_state.dataframes[new_final_df_name] = data
                        st.success("✔ Standarisasi selesai!")
                        st.balloons()
                        st.subheader("Hasil Akhir (Setelah Standarisasi)")
                        st.dataframe(data)
                    except Exception as e: st.error(f"Terjadi kesalahan saat analisis: {e}")

# --- HALAMAN UTAMA BARU: MODELING ---
elif st.session_state.sidebar_mode == "Modeling":
    st.header("🧠 Modeling: K-Means Clustering")

    available_dfs = list(st.session_state.dataframes.keys())
    if not available_dfs:
        st.warning("Tidak ada DataFrame yang tersedia. Siapkan data di mode sebelumnya terlebih dahulu.")
    else:
        model_df_name = st.selectbox(
            "1. Pilih DataFrame untuk dilatih:", 
            options=available_dfs, 
            help="Pilih DataFrame yang sudah bersih dan terstandarisasi untuk hasil terbaik."
        )

        if model_df_name:
            df_model = st.session_state.dataframes[model_df_name]
            
            X = df_model.select_dtypes(include=np.number)
            if X.empty:
                st.error("DataFrame yang dipilih tidak memiliki kolom numerik untuk clustering.")
            else:
                st.info(f"Fitur yang akan digunakan untuk clustering: **{', '.join(X.columns)}**")

                st.markdown("---")
                st.subheader("2. Tentukan Jumlah Cluster (k)")
                
                if st.button("Tampilkan Grafik Elbow untuk Rekomendasi 'k'"):
                    with st.spinner("Menghitung skor elbow..."):
                        try:
                            model = KMeans(init='k-means++', n_init='auto', random_state=42)
                            visualizer = KElbowVisualizer(model, k=(2,11), metric='distortion', timings=False)
                            visualizer.fit(X)
                            st.pyplot(visualizer.fig)
                            plt.clf() # Membersihkan figure setelah ditampilkan
                        except Exception as e:
                            st.error(f"Gagal membuat grafik Elbow: {e}")

                k_clusters = st.slider("Pilih jumlah cluster (k):", min_value=2, max_value=15, value=4)
                
                if st.button(f"Latih Model K-Means dengan k={k_clusters}"):
                    with st.spinner(f"Melatih model K-Means dengan {k_clusters} cluster..."):
                        try:
                            kmeans = KMeans(n_clusters=k_clusters, init='k-means++', n_init='auto', random_state=42)
                            clusters = kmeans.fit_predict(X)
                            
                            df_result = df_model.copy()
                            df_result['cluster'] = clusters
                            
                            pca = PCA(n_components=3, random_state=42)
                            components = pca.fit_transform(X)
                            df_pca = pd.DataFrame(data=components, columns=['PC1', 'PC2', 'PC3'], index=df_result.index)
                            df_pca['cluster'] = clusters
                            
                            score_s = silhouette_score(X, clusters)
                            score_c = calinski_harabasz_score(X, clusters)
                            score_d = davies_bouldin_score(X, clusters)
                            
                            st.session_state.modeling_results = {
                                "df_result": df_result,
                                "df_pca": df_pca,
                                "metrics": {"silhouette": score_s, "calinski": score_c, "davies": score_d},
                                "k": k_clusters
                            }
                            st.success(f"Model K-Means berhasil dilatih!")
                        except Exception as e:
                            st.error(f"Gagal melatih model: {e}")
                            st.session_state.modeling_results = {}

    if st.session_state.modeling_results:
        st.markdown("---")
        st.header(f"Hasil Clustering dengan k={st.session_state.modeling_results['k']}")

        st.subheader("Persebaran Anggota per Cluster")
        cluster_counts = st.session_state.modeling_results['df_result']['cluster'].value_counts().sort_index()
        st.dataframe(cluster_counts)
        st.bar_chart(cluster_counts)

        st.subheader("Visualisasi Persebaran Cluster (PCA)")
        plot_dim = st.radio("Pilih Tampilan Visualisasi:", ["2D", "3D"], horizontal=True)
        
        df_pca_result = st.session_state.modeling_results['df_pca']
        
        if plot_dim == "2D":
            fig = px.scatter(df_pca_result, x='PC1', y='PC2', color='cluster', 
                             title='Visualisasi Cluster 2D', hover_data=[df_pca_result.index])
        else:
            fig = px.scatter_3d(df_pca_result, x='PC1', y='PC2', z='PC3', color='cluster',
                               title='Visualisasi Cluster 3D', hover_data=[df_pca_result.index])
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Metrik Evaluasi Model")
        metrics = st.session_state.modeling_results['metrics']
        col1, col2, col3 = st.columns(3)
        col1.metric("Silhouette Score", f"{metrics['silhouette']:.4f}", help="Lebih tinggi lebih baik (rentang -1 s.d. 1).")
        col2.metric("Calinski-Harabasz Index", f"{metrics['calinski']:.2f}", help="Lebih tinggi lebih baik.")
        col3.metric("Davies-Bouldin Index", f"{metrics['davies']:.4f}", help="Lebih rendah lebih baik (mendekati 0).")

        st.markdown("---")
        final_df_name = st.text_input("Simpan hasil clustering sebagai DataFrame baru dengan nama:", f"clustered_{model_df_name}_{st.session_state.modeling_results['k']}k")
        if st.button("Simpan Hasil Clustering"):
            if not final_df_name: st.warning("Nama tidak boleh kosong.")
            elif final_df_name in st.session_state.dataframes: st.warning(f"Nama '{final_df_name}' sudah digunakan.")
            else:
                st.session_state.dataframes[final_df_name] = st.session_state.modeling_results['df_result']
                st.success(f"DataFrame '{final_df_name}' berhasil disimpan!")
else:
    st.info("👈 Silakan mulai dengan mengunggah file atau memilih mode di sidebar.")