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
import matplotlib.pyplot as plt
import seaborn as sns

#Melakukan Iterasi untuk mendapatkan nilai Cost
cost = {}
for k in range(2,10):
	kproto = KPrototypes(n_clusters = k,random_state=75)
	kproto.fit_predict(df_model, categorical=[0,1,2])
	cost[k]= kproto.cost_

#Visualisasi Elbow Plot
sns.pointplot(x=list(cost.keys()), y=list(cost.values()))
plt.show()

#Menyimpan model dengan jumlah cluster 5 berdasarkan Elbow Plot
import pickle
kproto = KPrototypes(n_clusters=5, random_state=75)
kproto = kproto.fit(df_model, categorical=[0,1,2])
pickle.dump(kproto, open('best_cluster.pkl', 'wb'))

#Menentukan segmen tiap pelanggan
clusters = kproto.predict(df_model, categorical=[0,1,2])
print('segmen_pelanggan: {}\n'.format(clusters))

#Menggabungkan data awal dan segmen pelanggan
df_final = df.copy()
df_final['cluster'] = clusters
print(df_final.head())

#Menampilkan data pelanggan berdasarkan cluster
for i in range (0,5):
	print('\nPelanggan cluster {}\n'.format(i))
	print(df_final[df_final['cluster']==i])

#Visualisasi box plot hasil clustering
for i in kolom_numerik:
	plt.figure(figsize=(6,4))
	ax = sns.boxplot(x = 'cluster',y = i, data = df_final)
	plt.title('\nBox Plot {}\n'.format(i), fontsize=12)
	plt.show()

#Visualisasi count plot hasil clustering
for i in kolom_kategorikal:
	plt.figure(figsize=(6, 4))
	ax = sns.countplot(data=df_final, x='cluster', hue=i)
	plt.title('\nCount Plot {}\n'.format(i), fontsize=12)
	ax.legend(loc="upper center")
	for p in ax.patches:
		ax.annotate(format(p.get_height(), '.0f'),
					(p.get_x() + p.get_width() / 2., p.get_height()),
					ha='center',
					va='center',
					xytext=(0, 10),
					textcoords='offset points')

	sns.despine(right=True, top=True, left=True)
	ax.axes.yaxis.set_visible(False)
	plt.show()

#Nama cluster
df_final['segmen'] = df_final['cluster'].map({
    0: 'Diamond Young Member',
    1: 'Diamond Senior Member',
    2: 'Silver Member',
    3: 'Gold Young Member',
    4: 'Gold Senior Member'
})


