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
    st.write(output.head(15))

#Mengelompokkan berdasarkan kolom product_category_name_english, kemudian menghitung banyaknya dari masing masing category_name_english, lalu mengurutkan dengan tipe Descending dan menampilkan 10 data pertama
bigten = df.groupby('product_category_name_english').size().sort_values(ascending=False).head(10)
#Mengelompokkan berdasarkan kolom product_category_name_english, kemudian menghitung banyaknya dari masing masing category_name_english, lalu mengurutkan dengan tipe Descending dan menampilkan 10 data terakhir
lowest10 = df.groupby('product_category_name_english').size().sort_values(ascending=False).tail(10)
#slicing df untuk mengambil informasi yang diperlukan untuk menjawab pertanyaan 2
df_2 = df[['product_category_name_english', 'order_purchase_timestamp']]
#mengekstrak order_purchase_timestamp hanya menjadi tanggal saja
df_2['tanggal']=df_2['order_purchase_timestamp'].apply(lambda x: re.findall(r'^.{0,10}', x)[0])
#Mencari jumlah pernjualan per produk per hari
df_2_clean = df_2.groupby(['product_category_name_english', 'tanggal']).size().reset_index(name='Count')
#slicing dataframe df_2_clean pada kolom product category name english yang bernilai bad_bath_table
final_2=df_2_clean[df_2_clean['product_category_name_english'].isin(['bed_bath_table'])]
#Mengubah tipe data kolom tanggal menjadi date
final_2['tanggal'] = pd.to_datetime(final_2['tanggal'])
# Mengatur kolom 'tanggal' sebagai indeks
final_2.set_index('tanggal', inplace=True)

# Menghitung penjualan per bulan
penjualan_per_bulan = final_2.resample('M').sum()
#reset index penjualan per bulan
penjualan_per_bulan.reset_index(inplace=True)

with tab2:
    #10 Kategori Produk Paling Diminati
    st.subheader('10 Kategori Produk Paling Diminati')
    # Membuat barplot
    plt.bar(bigten.index, bigten, color='skyblue')

    # Memberikan warna berbeda untuk nilai tertinggi
    max_value = bigten.max()
    max_index = bigten.idxmax()
    plt.bar(max_index, max_value, color='orange')

    # Set label dan judul
    plt.xlabel('Product Category Name')
    plt.ylabel('Count')
    plt.title('10 Kategori Produk Paling Diminati')

    # Menampilkan plot
    plt.xticks(rotation=90)  # Rotasi label x-axis supaya lebih enak dibaca
    plt.tight_layout()
    st.pyplot(plt)

    st.subheader('10 Kategori Produk Paling Kurang Diminati')
    # Membuat barplot
    plt.bar(lowest10.index, lowest10, color='skyblue')

    # Memberikan warna berbeda untuk nilai terendah
    min_value = lowest10.min()
    min_index = lowest10.idxmin()
    plt.bar(min_index, min_value, color='darkblue')

    # Set label dan judul
    plt.xlabel('Product Category Name')
    plt.ylabel('Count')
    plt.title('10 Kategori Produk Paling Kurang Diminati')

    # Menampilkan plot
    plt.xticks(rotation=90)  # Rotasi label x-axis supaya lebih enak dibaca
    plt.tight_layout()
    st.pyplot(plt)

    #tren produk paling diminati
    st.subheader('Grafik Penjualan Produk Paling Diminati: bed_bath_table')
    # Plot chart dari runtun waktu
    # plt.figure(figsize=(10, 6))
    plt.plot(penjualan_per_bulan['tanggal'], penjualan_per_bulan['Count'], marker='o', color='b', linestyle='-')

    # Set labels and title
    plt.xlabel('Tanggal')
    plt.ylabel('Count')
    plt.title('Grafik Penjualan Kategori Produk Paling Diminati (bed_bath_table)')

    # Show the plot
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)