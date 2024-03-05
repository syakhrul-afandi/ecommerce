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

#Membuat dataframe yang merupakan hasil dari join beberapa dataframe lain seperti products, order_items, orders, dan product_category
df = pd.merge(pd.merge(pd.merge(products, order_items, on='product_id', how='outer'), orders, on='order_id', how='left'), product_category, on='product_category_name', how='outer')
bigten = df.groupby('product_category_name_english').size().sort_values(ascending=False).head(10)
lowest10 = df.groupby('product_category_name_english').size().sort_values(ascending=True).head(10)
#slicing df untuk mengambil informasi yang diperlukan untuk menjawab pertanyaan 2
df_2 = df[['product_category_name_english', 'order_purchase_timestamp']]
#mengekstrak order_purchase_timestamp hanya menjadi tanggal saja
df_2['tanggal']=df_2['order_purchase_timestamp'].apply(lambda x: re.findall(r'^.{0,10}', x)[0])
#Mencari jumlah pernjualan per produk per hari
df_2_clean = df_2.groupby(['product_category_name_english', 'tanggal']).size().reset_index(name='Count')
final_2=df_2_clean[df_2_clean['product_category_name_english'].isin(['bed_bath_table'])]
for i in range(final_2.shape[0]):
  final_2['tanggal'].iloc[i] = datetime.strptime(final_2['tanggal'].iloc[i], '%Y-%m-%d').date()
final_2['tanggal'] = pd.to_datetime(final_2['tanggal'])
penjualan_per_bulan = final_2.groupby(final_2['tanggal'].dt.to_period('M')).sum()
st.subheader('10 Kategori Produk Terlaris')
#Membuat plot untuk 10 kategori terlaris
bigten.plot(kind='bar')
plt.xlabel('Kategori Produk Terjual')
plt.ylabel('Jumlah Produk')
st.pyplot(plt)

#Membuat plot untuk 10 kategori paling kurang diminati
st.subheader('10 Kategori Produk Paling Kurang Diminati')
lowest10.plot(kind='bar')
plt.xlabel('Kategori Produk')
plt.ylabel('Jumlah Produk Terjual')
st.pyplot(plt)


#Tren Produk Terlaris
st.subheader('Tren Penjualan Kategori Produk Terlaris')
plt.figure(figsize=(10, 6))
plt.plot(penjualan_per_bulan.index.astype(str), penjualan_per_bulan['Count'], marker='o', color='b', linestyle='-')
plt.xlabel('Bulan')
plt.ylabel('Jumlah Penjualan')
plt.title('Tren Penjualan Bulanan bed_bath_table')
plt.grid(True)
plt.xticks(rotation=45)  
plt.tight_layout()
st.pyplot(plt)

#Clustering
#Menyiapkan dataframe untuk clustering product
Analisis = pd.merge(df[['product_id', 'product_category_name_english', 'product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm', 'price']], df.groupby('product_id').size().reset_index(name='count_sold'), on='product_id', how='outer')
clustering = Analisis.groupby('product_category_name_english').agg({
    'product_weight_g': 'mean',
    'product_length_cm': 'mean',
    'product_height_cm': 'mean',
    'product_width_cm': 'mean',
    'price': 'mean',
    'count_sold': 'sum'
}).reset_index()
#Clustering Produk dengan fitur-fitur pada tabel di atas dengan menggunakan metode k-means menjadi 3 cluster, yakni produk paling laku, produk laku, dan produk kurang laku
np.random.seed(21060)
X = clustering[['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm', 'price', 'count_sold' ]]
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
clustering['cluster']=kmeans.predict(X)
output = clustering
output['Tingkat Kelakuan'] = np.where(output['cluster'] == 0, 'Laku',
                                    np.where(output['cluster'] == 1, 'Sangat Laku', 'Kurang Laku'))
output = output.drop(columns=['cluster'])

st.subheader('Data Penjualan Produk Dari 2016 Hingga 2018')
Analisis = Analisis.drop(columns=['product_id'])
st.sidebar.title('Filter Produk')
harga_min = st.sidebar.number_input('Harga Minimum:', value=0)
harga_max = st.sidebar.number_input('Harga Maksimum:', value=2000)
jenis_produk = st.sidebar.selectbox('Pilih Jenis Produk:', options=['Semua'] + Analisis['product_category_name_english'].unique().tolist())

# Filter data berdasarkan input dari sidebar
filtered_data = Analisis[(Analisis['price'] >= harga_min) & (Analisis['price'] <= harga_max)]
if jenis_produk != 'Semua':
    filtered_data = filtered_data[filtered_data['product_category_name_english'] == jenis_produk]

# Menentukan jumlah baris yang akan ditampilkan
if len(filtered_data) == len(Analisis):
    num_rows = 15
else:
    num_rows = len(filtered_data)

# Menampilkan data yang difilter
st.write(filtered_data.head(num_rows))






