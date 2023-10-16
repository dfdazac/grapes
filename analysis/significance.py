import scipy.stats as ss
import numpy as np
import scikit_posthocs as sp
from scipy.stats import wilcoxon

data = np.array([
    # Samples (datasets) ->                                   ↓ Treatments
    [74.93, 53.26, 53.73, 38.02, 41.06, 26.42, 29.91, 33.01],  # FastGCN
    [69.90, 67.85, 56.36, 81.88, 43.23, 16.02, 44.58, 67.72],  # LADIES
    [86.20, 77.63, 83.07, 80.50, 48.47, 33.13, 53.71, 59.57],  # GraphSAINT
    [86.42, 78.80, 89.76, 92.44, 48.21, 30.99, 65.95, -1.00],  # AS-GCN
    [87.29, 78.74, 90.11, 93.68, 47.33, 44.91, 64.54, 73.65]   # GFGS
])

no_asgcn = np.concatenate((data[:3], data[4:5]))
print(f'Means without AS-GCN={np.mean(no_asgcn, axis=1)}')
print(f'AS-GCN mean={np.mean(data[3, :7])}')

print(ss.friedmanchisquare(*data))
print(sp.posthoc_nemenyi_friedman(data.T))

ranks = np.argsort(np.argsort(-data, axis=0), axis=0) + 1
print(ranks)
print(np.sum(ranks, axis=1))
print(f'mean_ranks={np.mean(ranks, axis=1)}\n')

print(f'Wilcoxon signed-rank test with Bonferroni correction:')
models = ['FastGCN', 'LADIES', 'GraphSAINT', 'AS-GCN', 'GFGS']
gfgs_ranks = ranks[-1]
num_comparisons = len(models) - 1
for i in range(len(models) - 1):
    model_ranks = ranks[i]
    stat, p = wilcoxon(gfgs_ranks, model_ranks)
    # p = p * num_comparisons
    print(f'{models[i]}: stat={stat}, p={p}' + ('***' if p < 0.05/5.0 else ''))
