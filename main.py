from sklearn.cluster import OPTICS
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import time

df = pd.read_csv("Your file here")

# printa informações sobre o arquivo // print file information
print(df.describe(), df.head(), df.shape, df.info())

# cálculo do Nearest Neighbors para o eps // Nearest Neighbors calculation for eps
nbrs = NearestNeighbors(n_neighbors = 5).fit(df)
neigh_dist, neigh_ind = nbrs.kneighbors(df)
sort_neigh_dist = np.sort(neigh_dist, axis = 0)
k_dist = sort_neigh_dist[:, 4]

plt.plot(k_dist)
plt.axhline(y = 2.5, linewidth = 1, linestyle = "dashed", color = 'k')
plt.ylabel("k-NN distance")
plt.xlabel("Sorted observations")
plt.show()

# verificar o tempo de clusterização e clusteriza as amostras do documento // check the clustering time and cluster the document samples
start = time.time()
clusters = OPTICS(eps = 60, min_samples = 6).fit(df)
end = time.time()

# calcula a silhueta dos clusters // calculates the silhouette of the clusters
lable = clusters.labels_
print("silhouette_score:", metrics.silhouette_score(df, lable))

plot = sns.scatterplot(data = df, x = "X", y = "Y", hue = lable, legend = "full", palette = "deep")
sns.move_legend(plot, "upper right", bbox_to_anchor = (1.16, 1), title = "Clusters")

print("execution time:", end - start)
plt.show()