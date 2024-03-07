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

#import seluruh dataframe yang ada
order_items = pd.read_csv('https://raw.githubusercontent.com/syakhrul-afandi/ecommerce/main/Data%20E-commerce/olist_order_items_dataset.csv')
orders = pd.read_csv('https://raw.githubusercontent.com/syakhrul-afandi/ecommerce/main/Data%20E-commerce/olist_orders_dataset.csv')
products = pd.read_csv('https://raw.githubusercontent.com/syakhrul-afandi/ecommerce/main/Data%20E-commerce/olist_products_dataset.csv')
product_category = pd.read_csv('https://raw.githubusercontent.com/syakhrul-afandi/ecommerce/main/Data%20E-commerce/product_category_name_translation.csv')

#Membuat dataframe dari hasil join keempat dataframe sebelumnya
df=pd.merge(pd.merge(pd.merge(order_items, orders, on='order_id', how='left'), products, on='product_id', how='outer'), product_category, on = 'product_category_name', how='outer')

#CLUSTERING
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

tab1, tab2 = st.tabs(['Katalog Penjualan Produk', 'Ringkasan Penjualan Produk'])
