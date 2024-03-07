import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import re
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import streamlit as st

st.set_page_config(page_title="Analisis Data: E-commerce Public Dataset")
st.title('Analisis Data: E-commerce Public Dataset')
st.markdown("""
- **Nama:** Syakhrul Afandi
- **Email:** m008d4ky2924@bangkit.academy
- **ID Dicoding:** syakhrul_afandi_b8m2
            """)

#import seluruh dataframe yang dibutuhkan
order_items = pd.read_csv('https://raw.githubusercontent.com/syakhrul-afandi/ecommerce/main/Data%20E-commerce/olist_order_items_dataset.csv')
orders = pd.read_csv('https://raw.githubusercontent.com/syakhrul-afandi/ecommerce/main/Data%20E-commerce/olist_orders_dataset.csv')
products = pd.read_csv('https://raw.githubusercontent.com/syakhrul-afandi/ecommerce/main/Data%20E-commerce/olist_products_dataset.csv')
product_category = pd.read_csv('https://raw.githubusercontent.com/syakhrul-afandi/ecommerce/main/Data%20E-commerce/product_category_name_translation.csv')

products['product_category_name'].fillna('undefined', inplace=True)
products.fillna(products.mean(), inplace=True)

#Membuat dataframe dari hasil join keempat dataframe sebelumnya
df=pd.merge(pd.merge(pd.merge(order_items, orders, on='order_id', how='left'), products, on='product_id', how='outer'), product_category, on = 'product_category_name', how='outer')

#Menyiapkan dataframe untuk clustering product
Analisis = pd.merge(df[['product_id', 'product_category_name_english', 'product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm', 'price']], df.groupby('product_id').size().reset_index(name='count_sold'), on='product_id', how='outer')
#Mengelompokkan berdasarkan  kolom dengan agregasi mean dan sum pada kolom-kolom tertentu
clustering = Analisis.groupby('product_id').agg({
    'product_weight_g': 'mean',
    'product_length_cm': 'mean',
    'product_height_cm': 'mean',
    'product_width_cm': 'mean',
    'price': 'mean',
    'count_sold': 'sum'
}).reset_index()
#Join ke dataframe products dan product_category untuk mendapatkanproduct_category_name_english kembali
clustering = pd.merge(pd.merge(clustering, products[['product_id', 'product_category_name']], on='product_id', how='outer'), product_category, on='product_category_name', how='outer')
#Clustering Produk dengan fitur-fitur pada tabel di atas dengan menggunakan metode k-means menjadi 3 cluster, yakni produk paling laku, produk laku, dan produk kurang laku
np.random.seed(21060)
X = clustering[['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm', 'price', 'count_sold' ]]
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
clustering['cluster']=kmeans.predict(X)
#Memberi nilai Tingkat KElakuan berdasarkan cluster
output = clustering
output['Tingkat Penjualan'] = np.where(output['cluster'] == 0, 'Sangat Laku',
                                    np.where(output['cluster'] == 1, 'Kurang Laku', 'Cukup Laku'))
output = output.drop(columns=['cluster'])

tab1, tab2 = st.tabs(['Katalog Penjualan Produk', 'Ringkasan Penjualan Kategori Produk'])

with tab1:
    st.header('Katalog Penjualan Produk')
    st.sidebar.title('Filter Produk')
    harga_min = st.sidebar.number_input('Harga Minimum:', value=0)
    harga_max = st.sidebar.number_input('Harga Maksimum:', value=2000)
    jenis_produk = st.sidebar.selectbox('Pilih Jenis Produk:', options=['Semua'] + output['product_category_name_english'].unique().tolist())
    penjualan = st.sidebar.selectbox('Pilih Tingkat Penjualan:', options=['Semua']+output['Tingkat Penjualan'].unique().tolist())

    # Filter data berdasarkan input dari sidebar
    filtered_data = output[(output['price'] >= harga_min) & (output['price'] <= harga_max)]
    if jenis_produk != 'Semua' | penjualan != 'Semua':
        filtered_data = filtered_data[(filtered_data['product_category_name_english'] == jenis_produk) | (filtered_data['Tingkat Penjualan']==penjualan)]

    # Menentukan jumlah baris yang akan ditampilkan
    if len(filtered_data) == len(output):
        num_rows = 50
    else:
        num_rows = len(filtered_data)

    # Menampilkan data yang difilter
    st.write(filtered_data.head(num_rows))