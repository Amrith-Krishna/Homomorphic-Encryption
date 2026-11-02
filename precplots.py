import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

df = pd.read_csv("final_prec_v2.csv")

fig, ax = plt.subplots(figsize=(12, 6))

for col in [ "CKKS","CKKS log","CKKS+BFV", "BFV+BFV t_bits=16", "BFV+BFV t_bits=56"]:
    ax.plot(df["UB"], df[col], label=col, linewidth=1.5, alpha=0.85)

plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
ax.set_xlabel("Data Upper Bound (log10-scale)", fontsize=20)
ax.set_ylabel("Precision", fontsize=20)
ax.set_title("Algorithm Precision by Data Range", fontsize=20)
ax.legend(fontsize=18)
ax.grid(True, linestyle="--", alpha=0.4)
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))

plt.tight_layout()
plt.savefig("final_prec_plot.png", dpi=600)
plt.show()