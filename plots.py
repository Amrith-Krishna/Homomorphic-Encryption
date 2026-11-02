import matplotlib.pyplot as plt
import numpy as np

algorithms = [
    "Standard CKKS",
    "CKKS log",
    "CKKS+BFV",
    "BFV+BFV\nt_bits=16",
    "BFV+BFV\nt_bits=56"
]

means = [16.672,14.802,16.939,6.421,8.813]
sds   = [0.527,0.549,0.621,0.329,0.298]

x = np.arange(len(algorithms))

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(x, means, width=0.5, yerr=sds, capsize=10)

ax.set_xlabel('Algorithm', fontsize=20)
ax.set_ylabel('Time (ms)', fontsize=20)
ax.set_title('Algorithm Time Comparison', fontsize=22)
plt.yticks(fontsize=18)
ax.set_xticks(x)
ax.set_xticklabels(algorithms, fontsize=18)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('barplot.png', dpi=600)
plt.show()