import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv('seiseki.csv')

# 「国語ー英語」の散布図
sns_plot = sns.scatterplot(df['kokugo'], df['eigo'])
sns_plot.figure.savefig('PCA2dplot_kokugo_eigo.png')

# 「数学ー理科」の散布図
sns_plot = sns.scatterplot(df['sugaku'], df['rika'])
sns_plot.figure.savefig('PCA2dplot_sugaku_rika.png')

# 「数学ー体育」の散布図
sns_plot = sns.scatterplot(df['sugaku'], df['taiiku'])
sns_plot.figure.savefig('PCA2dplot_sugaku_taiiku.png')

# 相関行列の表示
df.corr()

# 散布図行列
sns_plot = sns.pairplot(df)
sns_plot.savefig('PCA_Scatter_plot_matrix.png')

# 「数学ー英語ー国語」の三次元散布図
fig = plt.figure()
ax = Axes3D(fig)

ax.set_xlabel('sugaku')
ax.set_ylabel('eigo')
ax.set_zlabel('kokugo')

ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_zlim(0, 100)

ax.scatter(df['sugaku'], df['eigo'], df['kokugo'])

plt.savefig('PCA_3dplot_sugaku_eigo_kokugo.png')

# 「数学ー音楽ー体育」の三次元散布図
fig = plt.figure()
ax = Axes3D(fig)

ax.set_xlabel('sugaku')
ax.set_ylabel('ongaku')
ax.set_zlabel('taiiku')

ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_zlim(0, 100)

ax.scatter(df['sugaku'], df['ongaku'], df['taiiku'])

plt.savefig('PCA_3dplot_sugaku_ongaku_taiiku.png')

# 主成分分析
ss = StandardScaler()
df_s = ss.fit_transform(df)
pca = PCA()
pca.fit(df_s)
# 主成分得点の散布図
feature = pca.transform(df_s)

feature = pd.DataFrame(feature, columns=['PC{}'.format(i + 1) for i in range(len(df.columns))])
sns_plot = sns.scatterplot(-feature['PC1'], -feature['PC2'])
sns_plot.figure.savefig('pca_feature.png')

# 因子負荷量の散布図
components = pca.components_

plt.figure()

plt.xlabel('PC1')
plt.ylabel('PC2')

plt.xlim(-1, 1)
plt.ylim(-1, 1)

line = np.arange(-1, 1, 0.01)

plt.plot(line, [0] * len(line), ls='--', linewidth=1.0, color='0.5')
plt.plot([0] * len(line), line, ls='--', linewidth=1.0, color='0.5')

colors = [plt.cm.hsv(i / len(df.columns)) for i in range(len(df.columns))]

for i, column in enumerate(df.columns):
    plt.scatter(-components[0][i], -components[1][i], c = colors[i], label=column)

plt.legend(loc = 'upper left')
plt.savefig('pca_components.png')

# Biplot
plt.figure()

plt.xlabel('PC1')
plt.ylabel('PC2')

plt.scatter(-feature['PC1'], -feature['PC2'], marker='.')

for i, name in enumerate(df.columns):
    plt.arrow(0, 0, -components[0][i] * 2.5, -components[1][i] * 2.5, color='r')
    plt.text(-components[0][i] * 2.5 * 1.05, -components[1][i] * 2.5 * 1.05, name, color='r')

plt.savefig('pca_biplot.png')
