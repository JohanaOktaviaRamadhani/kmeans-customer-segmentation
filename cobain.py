import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from yellowbrick.cluster import KElbowVisualizer
import os

def preprocess_and_cluster(df_orders, df_order_items, df_products, df_customers):

    # --- Filtering dan pembersihan data ---
    df_orders = df_orders.dropna(subset=['order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date'])
    df_orders.reset_index(drop=True, inplace=True)

    index_to_drop = df_orders[df_orders['order_status'] == 'canceled'].index[0]
    df_orders = df_orders.drop(index_to_drop)

    df_orders = df_orders[df_orders['order_status'] == 'delivered']
    df_order_items = df_order_items.copy()
    df_order_items['shipping_limit_date'] = pd.to_datetime(df_order_items['shipping_limit_date'])
    df_products = df_products.dropna().copy()
    df_products['product_volume_cm3'] = df_products['product_length_cm'] * df_products['product_height_cm'] * df_products['product_width_cm']

    product_cols = [
        'product_id', 'product_category_name', 'product_photos_qty',
        'product_name_lenght', 'product_description_lenght', 'product_weight_g', 'product_volume_cm3'
    ]
    df_products = df_products[product_cols]

    # Merge datasets
    df_merge = pd.merge(df_orders, df_order_items, on='order_id', how='inner')
    df_merge = pd.merge(df_merge, df_products, on='product_id', how='inner')
    df_full = pd.merge(df_merge, df_customers, on='customer_id', how='inner')

    columns_to_drop = [
        'order_status', 'order_approved_at', 'order_delivered_carrier_date', 'order_estimated_delivery_date',
        'seller_id', 'shipping_limit_date', 'product_photos_qty', 'product_name_lenght',
        'product_description_lenght', 'product_weight_g', 'customer_unique_id',
        'customer_zip_code_prefix', 'customer_city', 'customer_state'
    ]
    df_clean = df_full.drop(columns=columns_to_drop)
    df = df_clean

    # Agregasi data per customer_id
    agg_data = df.groupby('customer_id').agg(
        num_orders=('order_id', pd.Series.nunique),
        num_items=('order_item_id', 'sum'),
        total_spent=('price', 'sum'),
        avg_spent_per_order=('price', lambda x: x.sum() / x.nunique()),
        avg_freight_value=('freight_value', 'mean'),
        num_categories=('product_category_name', pd.Series.nunique),
        first_purchase=('order_purchase_timestamp', 'min'),
        last_purchase=('order_purchase_timestamp', 'max')
    ).reset_index()

    agg_data['first_purchase'] = pd.to_datetime(agg_data['first_purchase'])
    agg_data['last_purchase'] = pd.to_datetime(agg_data['last_purchase'])

    # Hitung rentang waktu pembelian dan rata-rata interval
    agg_data['purchase_span_days'] = (agg_data['last_purchase'] - agg_data['first_purchase']).dt.days
    agg_data['avg_order_interval'] = agg_data['purchase_span_days'] / (agg_data['num_orders'] - 1)

    # Handle div by zero
    agg_data['avg_order_interval'] = agg_data['avg_order_interval'].replace([float('inf'), -float('inf')], 0).fillna(0)

    # --- Scaling ---
    features_to_scale = [
        'num_orders', 'num_items', 'total_spent', 'avg_spent_per_order',
        'avg_freight_value', 'num_categories', 'purchase_span_days', 'avg_order_interval'
    ]
    scaler = StandardScaler()
    agg_data_scaled = agg_data.copy()
    agg_data_scaled[features_to_scale] = scaler.fit_transform(agg_data[features_to_scale])
    X = agg_data_scaled[features_to_scale]

    # --- Tentukan k optimal ---
    model = KMeans(random_state=42, n_init=10)
    visualizer = KElbowVisualizer(model, k=(1, 11), random_state=42)
    visualizer.fit(X)

    # --- Clustering final ---
    kmeans = KMeans(n_clusters=4, random_state=42)
    cluster_labels = kmeans.fit_predict(X)

    # --- Evaluasi cluster ---
    sample_size = min(5000, len(X))
    silhouette = silhouette_score(X, cluster_labels, sample_size=sample_size, random_state=42)
    calinski = calinski_harabasz_score(X, cluster_labels)
    davies = davies_bouldin_score(X, cluster_labels)

    agg_data['cluster'] = cluster_labels
    agg_data_scaled['cluster'] = cluster_labels

    return dict(
        agg_data=agg_data,
        agg_data_scaled=agg_data_scaled,
        X=X,
        features=features_to_scale,
        cluster_labels=cluster_labels,
        optimal_k=4,
        silhouette=silhouette,
        calinski=calinski,
        davies=davies,
        elbow_fig=visualizer.fig,
        kmeans_model=kmeans,
        scaler=scaler  # Mengembalikan scaler untuk inverse transform
    )

if 'page' not in st.session_state:
    st.session_state.page = 'hasil'

if 'stage' not in st.session_state:
    st.session_state.stage = 0
    st.session_state.wrangling_done = False
    st.session_state.preprocessing_done = False
    st.session_state.modeling_done = False
    st.session_state.evaluation_done = False
    st.session_state.data = {}

def show_hasil_page():
    st.title('🚀 Hasil Clustering Segmentasi Pelanggan (Otomatis)')
    st.markdown("Halaman ini menampilkan hasil clustering otomatis pada dataset bawaan.")

    @st.cache_data
    def load_data(file_path):
        if not os.path.exists(file_path):
            st.error(f"File tidak ditemukan: `{file_path}`. Pastikan file ada di direktori yang benar.")
            st.info("Struktur folder:\n- app.py\n- dataset/\n  - E-Commerce Public Dataset/\n    - customers_dataset.csv\n    - ...")
            return None
        return pd.read_csv(file_path)

    base_path = 'dataset/E-Commerce Public Dataset/'

    with st.spinner("Memuat dan memproses data... Ini mungkin memakan waktu beberapa saat."):
        df_customers = load_data(os.path.join(base_path, 'customers_dataset.csv'))
        df_order_items = load_data(os.path.join(base_path, 'order_items_dataset.csv'))
        df_orders = load_data(os.path.join(base_path, 'orders_dataset.csv'))
        df_products = load_data(os.path.join(base_path, 'products_dataset.csv'))
        if any(df is None for df in [df_orders, df_order_items, df_products, df_customers]):
            st.stop()
        
        # Panggil helper preprocessing & clustering yang sudah distandarisasi
        hasil = preprocess_and_cluster(df_orders, df_order_items, df_products, df_customers)

    st.success("Proses selesai!")
    st.divider()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Visualisasi Segmen Pelanggan (Fitur Paling Dominan)")
        
        # --- LOGIKA BARU: Tentukan fitur paling dominan ---
        centroids_scaled = hasil['kmeans_model'].cluster_centers_
        # Hitung varians dari centroid untuk setiap fitur
        centroid_variances = np.var(centroids_scaled, axis=0)
        # Dapatkan indeks dari 2 fitur dengan varians tertinggi
        dominant_feature_indices = np.argsort(centroid_variances)[-2:]
        
        # Dapatkan nama fitur dari indeks tersebut
        features = hasil['features']
        x_axis = features[dominant_feature_indices[1]] # Fitur paling dominan
        y_axis = features[dominant_feature_indices[0]] # Fitur dominan kedua
        
        st.info(f"Visualisasi otomatis berdasarkan fitur paling dominan: **{x_axis.replace('_',' ').title()}** dan **{y_axis.replace('_',' ').title()}**.")

        # Ambil centroids dan kembalikan ke skala asli
        scaler = hasil['scaler']
        centroids_original = scaler.inverse_transform(centroids_scaled)
        centroids_df = pd.DataFrame(centroids_original, columns=features)

        # Buat plot
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.set_style("whitegrid")
        
        # Plot data points
        sns.scatterplot(
            data=hasil['agg_data'], x=x_axis, y=y_axis, hue='cluster', 
            palette='viridis', s=50, alpha=0.7, ax=ax, legend='full'
        )
        
        # Plot centroids
        ax.scatter(
            centroids_df[x_axis], centroids_df[y_axis], 
            marker='*', s=400, c='red', label='Centroid', edgecolor='black'
        )

        ax.set_title(f'Visualisasi Cluster: {y_axis.replace("_", " ").title()} vs {x_axis.replace("_", " ").title()}', fontsize=16)
        ax.set_xlabel(x_axis.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(y_axis.replace('_', ' ').title(), fontsize=12)
        ax.legend(title='Cluster')
        ax.grid(True)
        
        st.pyplot(fig)


    with col2:
        st.subheader("Metrik Evaluasi")
        st.metric(label="Silhouette Score", value=f"{hasil['silhouette']:.4f}")
        st.metric(label="Calinski-Harabasz", value=f"{hasil['calinski']:.2f}")
        st.metric(label="Davies-Bouldin", value=f"{hasil['davies']:.4f}")
        st.info(f"Metrik dihitung dengan **k={hasil['optimal_k']}** cluster.")

    st.subheader("Profil Rata-rata per Segmen")
    cluster_summary = hasil['agg_data'].groupby('cluster')[hasil['features']].mean().round(2)
    st.dataframe(cluster_summary)
    st.divider()
    st.header("Coba Model Secara Interaktif")
    st.write("Klik tombol di bawah untuk masuk ke mode interaktif dengan data Anda sendiri.")

    if st.button("➡️ Coba Model Interaktif", type="primary"):
        st.session_state.page = 'interaktif'
        st.session_state.stage = 0
        st.session_state.wrangling_done = False
        st.session_state.preprocessing_done = False
        st.session_state.modeling_done = False
        st.session_state.evaluation_done = False
        st.session_state.data = {}
        st.rerun()

def show_interactive_page():
    def restart_interactive_app():
        st.session_state.stage = 0
        st.session_state.wrangling_done = False
        st.session_state.preprocessing_done = False
        st.session_state.modeling_done = False
        st.session_state.evaluation_done = False
        st.session_state.data = {}
        st.rerun()

    st.title("📊 Aplikasi Segmentasi Pelanggan (Interaktif)")
    st.markdown("---")

    with st.sidebar:
        st.header("Tahapan Analisis")
        stages = ["Data Wrangling", "Preprocessing", "Modeling", "Hasil & Evaluasi"]
        status_flags = [st.session_state.wrangling_done, st.session_state.preprocessing_done, st.session_state.modeling_done, st.session_state.evaluation_done]
        for i, stage in enumerate(stages):
            st.markdown(f"✅ **{i+1}. {stage}**" if status_flags[i] else f"◻️ {i+1}. {stage}")
        
        st.divider()
        if st.button("Mulai Ulang Analisis Interaktif"):
            restart_interactive_app()
        if st.button("⬅️ Kembali ke Halaman Hasil"):
            st.session_state.page = 'hasil'
            st.rerun()

    # TAHAP 0: UPLOAD DATA
    if st.session_state.stage == 0:
        st.header("Langkah 1: Upload Dataset")
        st.info("Unggah 4 file CSV: customers_dataset.csv, orders_dataset.csv, order_items_dataset.csv, products_dataset.csv.")
        uploaded_files = st.file_uploader("Upload file CSV di sini", type=['csv'], accept_multiple_files=True)
        expected_names = {'orders_dataset.csv', 'order_items_dataset.csv', 'products_dataset.csv', 'customers_dataset.csv'}
        uploaded_names = {f.name for f in uploaded_files}

        if expected_names.issubset(uploaded_names):
            st.success("Semua file yang dibutuhkan telah diunggah!")
            if st.button("Jalankan Data Wrangling", type="primary"):
                try:
                    file_map = {f.name: f for f in uploaded_files}
                    st.session_state.data.update({
                        'orders': pd.read_csv(file_map['orders_dataset.csv']),
                        'order_items': pd.read_csv(file_map['order_items_dataset.csv']),
                        'products': pd.read_csv(file_map['products_dataset.csv']),
                        'customers': pd.read_csv(file_map['customers_dataset.csv'])
                    })
                    st.session_state.wrangling_done = True
                    st.session_state.stage = 1
                    st.rerun()
                except Exception as e:
                    st.error(f"Terjadi error saat memproses file: {e}")
        elif uploaded_files:
            st.warning(f"File belum lengkap. File yang dibutuhkan: **{', '.join(expected_names - uploaded_names)}**")

    # TAHAP 1: PREPROCESSING
    elif st.session_state.stage == 1:
        st.header("Langkah 2: Preprocessing")
        st.markdown("Data akan digabungkan, fitur dibuat, dan dilakukan standarisasi.")
        if st.button("Jalankan Preprocessing", type="primary"):
            with st.spinner("Melakukan preprocessing dan feature engineering..."):
                data = st.session_state.data
                hasil = preprocess_and_cluster(data['orders'], data['order_items'], data['products'], data['customers'])
                st.session_state.data.update(hasil)
                st.session_state.preprocessing_done = True
                st.session_state.stage = 2
                st.rerun()
        
        if st.session_state.preprocessing_done:
            st.success("Preprocessing selesai!")
            st.subheader("Data Agregasi (5 baris pertama)")
            st.dataframe(st.session_state.data['agg_data'].head())
            if st.button("Lanjut ke Modeling"):
                st.rerun()

    # TAHAP 2: MODELING
    elif st.session_state.stage == 2:
        st.header("Langkah 3: Modeling (K-Means Clustering)")
        st.markdown("Menentukan jumlah cluster optimal menggunakan Elbow Method dan menjalankan K-Means.")
        if not st.session_state.modeling_done:
            st.pyplot(st.session_state.data['elbow_fig'])
            st.success(f"Ditemukan jumlah cluster optimal: **{st.session_state.data['optimal_k']}**.")
            st.session_state.modeling_done = True
        
        if st.session_state.modeling_done:
            if st.button("Lanjut ke Hasil & Evaluasi"):
                st.session_state.stage = 3
                st.rerun()

    # TAHAP 3: HASIL & EVALUASI
    elif st.session_state.stage == 3:
        st.header("Langkah 4: Hasil & Evaluasi")
        st.markdown("Analisis segmen dan evaluasi kualitas clustering.")
        
        if not st.session_state.evaluation_done:
            data = st.session_state.data
            
            st.subheader("Profil Rata-Rata per Segmen")
            cluster_summary = data['agg_data'].groupby('cluster')[data['features']].mean().round(2)
            st.dataframe(cluster_summary)

            st.subheader("Evaluasi Kualitas Model")
            col1, col2, col3 = st.columns(3)
            col1.metric("Silhouette Score", f"{data['silhouette']:.4f}")
            col2.metric("Calinski-Harabasz", f"{data['calinski']:.2f}")
            col3.metric("Davies-Bouldin", f"{data['davies']:.4f}")

            # --- BLOK VISUALISASI INTERAKTIF ---
            st.subheader("Visualisasi Interaktif Segmentasi Pelanggan")
            st.markdown("Pilih dua fitur untuk memvisualisasikan sebaran cluster.")

            col_x, col_y = st.columns(2)
            features_list = data['features']
            # Default ke fitur yang paling umum untuk segmentasi
            default_x = 'total_spent' if 'total_spent' in features_list else features_list[0]
            default_y = 'num_orders' if 'num_orders' in features_list else features_list[1]
            
            x_axis = default_x
            y_axis = default_y

            # Ambil centroids dan kembalikan ke skala asli
            centroids_scaled = data['kmeans_model'].cluster_centers_
            scaler = data['scaler']
            centroids_original = scaler.inverse_transform(centroids_scaled)
            centroids_df = pd.DataFrame(centroids_original, columns=features_list)

            # Buat plot
            fig_scatter, ax = plt.subplots(figsize=(12, 8))
            
            # Plot data points
            sns.scatterplot(
                data=data['agg_data'], x=x_axis, y=y_axis, hue='cluster', 
                palette='viridis', s=50, alpha=0.7, ax=ax, legend='full'
            )
            
            # Plot centroids
            ax.scatter(
                centroids_df[x_axis], centroids_df[y_axis], 
                marker='*', s=400, c='red', label='Centroid', edgecolor='black'
            )

            ax.set_title(f'Visualisasi Cluster: {y_axis.replace("_", " ").title()} vs {x_axis.replace("_", " ").title()}', fontsize=16)
            ax.set_xlabel(x_axis.replace('_', ' ').title(), fontsize=12)
            ax.set_ylabel(y_axis.replace('_', ' ').title(), fontsize=12)
            ax.legend(title='Cluster')
            ax.grid(True)
            
            st.pyplot(fig_scatter)
            # --- AKHIR BLOK VISUALISASI INTERAKTIF ---

            st.session_state.evaluation_done = True
            st.success("Analisis selesai! 🎉")

if st.session_state.page == 'hasil':
    show_hasil_page()
elif st.session_state.page == 'interaktif':
    show_interactive_page()
