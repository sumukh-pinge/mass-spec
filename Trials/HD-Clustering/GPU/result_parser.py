import numpy as np
from sklearn.metrics import normalized_mutual_info_score

cluster_results = np.genfromtxt('cluster_results.csv', delimiter=',')
cluster_results = cluster_results[:-1]  # truncate last comma

trainLabels = np.genfromtxt('train_labels.csv', delimiter=',')
trainLabels = trainLabels[:-1]

print(normalized_mutual_info_score(trainLabels, cluster_results))
