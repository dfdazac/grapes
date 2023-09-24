import scipy.stats as ss
import numpy as np
import scikit_posthocs as sp
from scipy.stats import wilcoxon

data = np.array([
    # Samples (datasets) ->                                    ↓ Treatments
    [74.93, 53.26, 53.73, 38.02, 41.06, 26.42, 29.91, 33.01],  # FastGCN
    [69.90, 67.85, 56.36, 81.88, 43.23, 16.02, 44.58, 67.72],  # LADIES
    [86.20, 77.63, 83.07, 80.50, 44.69, 33.13, 53.71, 59.57],  # GraphSAINT
    [80.70, 70.42, 79.22, 94.83, 51.32, 33.79, 69.38, 75.12],  # GAS
    [86.42, 78.80, 89.76, 92.44, 48.21, 30.99, 65.95, -1.00],  # AS-GCN
    [87.29, 75.17, 90.11, 93.68, 47.33, 44.91, 64.54, 73.65]   # GFGS
])

print(ss.friedmanchisquare(*data))
print(sp.posthoc_nemenyi_friedman(data.T) <= 0.05)

ranks = np.argsort(np.argsort(-data, axis=0), axis=0) + 1
print(f'mean_ranks={np.mean(ranks, axis=1)}\n')

print(f'Wilcoxon signed-rank test with Bonferroni correction:')
models = ['FastGCN', 'LADIES', 'GraphSAINT', 'GAS', 'AS-GCN', 'GFGS']
gfgs_ranks = ranks[-1]
num_comparisons = len(models) - 1
for i in range(len(models) - 1):
    model_ranks = ranks[i]
    stat, p = wilcoxon(gfgs_ranks, model_ranks)
    p = p * num_comparisons
    print(f'{models[i]}: stat={stat}, p={p}' + ('***' if p < 0.05 else ''))
