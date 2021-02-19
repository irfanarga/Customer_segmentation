import pandas as pd

#Panggil dataset
df = pd.read_csv('data_customer.txt', sep='\t')
print(df.head())
print(df.info())

import matplotlib.pyplot as plt
import seaborn as sns

#Eksplorasi data numerik
def observasi_num(features):
	fig, axs = plt.subplots(2, 2, figsize=(10, 9))
	for i, kol in enumerate(features):
		sns.boxplot(df[kol], ax = axs[i][0])
		sns.distplot(df[kol], ax = axs[i][1])
		axs[i][0].set_title('mean = %.2f\n median = %.2f\n std = %.2f'%(df[kol].mean(), df[kol].median(), df[kol].std()))
	plt.setp(axs)
	plt.tight_layout()
	plt.show()

#Memanggil fungsi
kolom_numerik = ['Umur', 'NilaiBelanjaSetahun']
observasi_num(kolom_numerik)

#Eksplorasi data kategorik
kolom_kategorikal = ['Jenis Kelamin','Profesi','Tipe Residen']
fig, axs = plt.subplots(3,1,figsize=(7,10))
for i, kol in enumerate(kolom_kategorikal):
	sns.countplot(df[kol], order = df[kol].value_counts().index, ax = axs[i])
	axs[i].set_title('\nCount Plot %s\n'%(kol), fontsize=15)

	for p in axs[i].patches:
		axs[i].annotate(format(p.get_height(), '.0f'),
						(p.get_x() + p.get_width() / 2., p.get_height()),
						ha='center',
						va='center',
						xytext=(0, 10),
						textcoords='offset points')

	sns.despine(right=True, top=True, left=True)
	axs[i].axes.yaxis.set_visible(False)
	plt.setp(axs)
	plt.tight_layout()

plt.show()

#Standarisasi nilai numerik
from sklearn.preprocessing import StandardScaler

#Statistik sebelum standarisasi
print('Statistik Sebelum Standardisasi:')
print(df[kolom_numerik].describe().round(1))

#Standarisasi numerik
df_standar = StandardScaler().fit_transform(df[kolom_numerik])

#Membuat dataframe
df_standar = pd.DataFrame(data=df_standar, index=df.index, columns=df[kolom_numerik].columns)

#Statistik setelah standarisasi
print('\nStatistik hasil standardisasi:')
print(df_standar.describe().round(0))

#Konversi data kategorik ke data numerik
from sklearn.preprocessing import LabelEncoder

#Membuat salinan dataframe
df_encode = df[kolom_kategorikal].copy()

#Menerapkan Label Encoder ke semua kolom kategorik
for col in kolom_kategorikal:
	df_encode[col] = LabelEncoder().fit_transform(df_encode[col])
print(df_encode.head())

#Menggabungkan dataframe
df_model = df_encode.merge(df_standar, left_index=True, right_index=True, how='left')
print(df_model.head())
