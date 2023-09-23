import scipy.stats as ss
import numpy as np
import scikit_posthocs as sp


data = np.array([
    # Samples (datasets) ->                                    ↓ Treatments
    [74.93, 53.26, 53.73, 38.02, 41.06, 26.42, 29.91, 33.01],  # FastGCN
    [69.90, 67.85, 56.36, 81.88, 43.23, 16.02, 44.58, 67.72],  # LADIES
    [86.20, 77.63, 83.07, 80.50, 44.69, 33.13, 53.71, 59.57],  # GraphSAINT
    [80.70, 70.42, 79.22, 94.83, 51.32, 33.79, 69.38, 75.12],  # GAS
    [86.42, 78.80, 89.76, 92.44, 48.21, 30.99, 65.95, 1000.],  # AS-GCN
    [87.29, 75.17, 90.11, 93.68, 47.33, 44.91, 64.54, 73.65]   # GFGS
])

print(ss.friedmanchisquare(*data))
print(sp.posthoc_nemenyi_friedman(data.T) <= 0.05)

ranks = np.argsort(np.argsort(-data, axis=0), axis=0) + 1
print(f'mean_ranks={np.mean(ranks, axis=1)}')
