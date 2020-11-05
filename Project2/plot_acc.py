import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

plt.rcParams.update({'font.size': 12})

penalties = np.logspace(-2,0,11)
etas = np.logspace(-3,-1,11)
accuracy_map = np.load("accuracy_map.npy")

etas = etas[2:]
accuracy_test = accuracy_map[:,2:,0]

accuracy_list = []
for i in range(11):
    accuracy_list.append([float("%.2f" %k) for k in accuracy_test[i, :]])

print(accuracy_list)

heatmap = sb.heatmap(accuracy_test,cmap="viridis",
                              xticklabels=["%.3f" %i for i in etas],
                              yticklabels=["%.3f" %j for j in penalties],
                              cbar_kws={'label': 'Accuracy'},
                              fmt = ".5",
                              annot = accuracy_list)
heatmap.set_xlabel(r"$\eta$")
heatmap.set_ylabel(r"$\lambda$")
heatmap.invert_yaxis()
heatmap.set_title("Accuracy on handwritten digits")
fig = heatmap.get_figure()
plt.show()
fig.savefig("./figures/MNIST_heatmap.pdf", bbox_inches='tight',
                                            pad_inches=0.1)
