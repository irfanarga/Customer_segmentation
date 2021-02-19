import pandas as pd

#Panggil dataset
df = pd.read_csv('data_customer.txt', sep='\t')
kolom_numerik = ['Umur', 'NilaiBelanjaSetahun']
kolom_kategorikal = ['Jenis Kelamin','Profesi','Tipe Residen']

#Standarisasi nilai numerik
from sklearn.preprocessing import StandardScaler

#Standarisasi numerik
df_standar = StandardScaler().fit_transform(df[kolom_numerik])

#Membuat dataframe hasil standarisasi
df_standar = pd.DataFrame(data=df_standar, index=df.index, columns=df[kolom_numerik].columns)

#Konversi data kategorik ke data numerik
from sklearn.preprocessing import LabelEncoder

#Membuat salinan dataframe
df_encode = df[kolom_kategorikal].copy()

#Menerapkan Label Encoder ke semua kolom kategorik
for col in kolom_kategorikal:
	df_encode[col] = LabelEncoder().fit_transform(df_encode[col])

#Menggabungkan dataframe
df_model = df_encode.merge(df_standar, left_index=True, right_index=True, how='left')

from kmodes.kprototypes import KPrototypes

#Menyimpan model dengan jumlah cluster 5 berdasarkan Elbow Plot
import pickle
kproto = KPrototypes(n_clusters=5, random_state=75)
kproto = kproto.fit(df_model, categorical=[0,1,2])
pickle.dump(kproto, open('best_cluster.pkl', 'wb'))

#Menentukan segmen tiap pelanggan
clusters = kproto.predict(df_model, categorical=[0,1,2])

#Menggabungkan data awal dan segmen pelanggan
df_final = df.copy()
df_final['cluster'] = clusters

#Nama cluster
df_final['segmen'] = df_final['cluster'].map({
	0: 'Diamond Young Member',
	1: 'Diamond Senior Member',
	2: 'Silver Member',
	3: 'Gold Young Member',
	4: 'Gold Senior Member'
})

############################################### Menambah Data Masukan Baru ########################################
#Contoh data baru
data = [{
	'Customer_ID': 'CUST-100',
	'Nama Pelanggan': 'Joko',
	'Jenis Kelamin': 'Pria',
	'Umur': 45,
	'Profesi': 'Wiraswasta',
	'Tipe Residen': 'Cluster',
	'NilaiBelanjaSetahun': 8230000

}]

new_df = pd.DataFrame(data)
print(new_df)

#Membuat fungsi data pra pemrosesan
def data_preprocess(data):
	# Konversi Kategorikal data
	kolom_kategorikal = ['Jenis Kelamin', 'Profesi', 'Tipe Residen']

	df_encode = data[kolom_kategorikal].copy()

	## Jenis Kelamin
	df_encode['Jenis Kelamin'] = df_encode['Jenis Kelamin'].map({
		'Pria': 0,
		'Wanita': 1
	})

	## Profesi
	df_encode['Profesi'] = df_encode['Profesi'].map({
		'Ibu Rumah Tangga': 0,
		'Mahasiswa': 1,
		'Pelajar': 2,
		'Professional': 3,
		'Wiraswasta': 4
	})

	## Tipe Residen
	df_encode['Tipe Residen'] = df_encode['Tipe Residen'].map({
		'Cluster': 0,
		'Sector': 1
	})

	# Standardisasi Numerical Data
	kolom_numerik = ['Umur', 'NilaiBelanjaSetahun']
	df_std = data[kolom_numerik].copy()

	## Standardisasi Kolom Umur
	df_std['Umur'] = (df_std['Umur'] - 37.5) / 14.7

	## Standardisasi Kolom Nilai Belanja Setahun
	df_std['NilaiBelanjaSetahun'] = (df_std['NilaiBelanjaSetahun'] - 7069874.8) / 2590619.0

	# Menggabungkan Kategorikal dan numerikal data
	df_model = df_encode.merge(df_std, left_index=True,
							   right_index=True, how='left')

	return df_model

new_df_model = data_preprocess(new_df)
print(new_df_model)

#Memanggil model dan melakukan prediksi
def modelling(data):
	# Memanggil Model
	kpoto = pickle.load(open('best_cluster.pkl', 'rb'))
	# Melakukan Prediksi
	clusters = kpoto.predict(data, categorical=[0, 1, 2])
	return clusters

# Menjalankan Fungsi
clusters = modelling(new_df_model)
print(clusters)

#Menamakan segmen
def menamakan_segmen(data_asli, clusters):
	# Menggabungkan cluster dan data asli
	final_df = data_asli.copy()
	final_df['cluster'] = clusters
	# Menamakan segmen
	final_df['segmen'] = final_df['cluster'].map({
		0: 'Diamond Young Member',
		1: 'Diamond Senior Member',
		2: 'Silver Students',
		3: 'Gold Young Member',
		4: 'Gold Senior Member'
	})
	return final_df

# Menjalankan Fungsi
new_final_df = menamakan_segmen(new_df, clusters)
print(new_final_df)
