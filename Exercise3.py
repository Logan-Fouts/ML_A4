import pandas as pd
from Functions import *

# Load Datasets
subset_size = 1000

bc = datasets.load_breast_cancer()
(bcX,index) = np.unique(bc.data,axis=0,return_index=True)
bcTarget = bc.target[index]
bcNames = bc.target_names

data = pd.read_excel("Data/Raisin_Dataset.xlsx")
raisinX = data.drop("Class", axis=1).values
raisinTarget = data["Class"].values

data = pd.read_excel("Data/Dry_Bean_Dataset.xlsx")
beanSubset = data.sample(n=subset_size, random_state=42)
beanX = beanSubset.drop("Class", axis=1).values
beanTarget = beanSubset["Class"].values

# Run bkMeans for each dataset
k = 2
iterations = 100
bc_cluster_indices = bkmeans(bcX, k, iterations)

k = 2
iterations = 100
raisin_cluster_indices = bkmeans(raisinX, k, iterations)

k = 7
iterations = 10
bean_cluster_indices = bkmeans(beanX, k, iterations)

# Run Sammon Mapping for each dataset
iter = 100
e = 1e-7
a = 100
bcResult = sammon(bcX, iter, e, a)

iter = 100
e = 1e-7
a = 100
raisin_result = sammon(raisinX, iter, e, a)

iter = 100
e = 1e-7
a = 100
beanResult = sammon(beanX, iter, e, a)

fig, axs = plt.subplots(2, 3, figsize=(15, 10))

axs[0, 0].scatter(bcX[:, 0], bcX[:, 1], c=bc_cluster_indices)
axs[0, 0].set_xlabel('X')
axs[0, 0].set_ylabel('Y')
axs[0, 0].set_title('Breast Cancer bkmeans Clustering')
axs[0, 0].legend()

axs[0, 1].scatter(raisinX[:, 0], raisinX[:, 1], c=raisin_cluster_indices)
axs[0, 1].set_xlabel('X')
axs[0, 1].set_ylabel('Y')
axs[0, 1].set_title('Rasin bkMeans Clustering')
axs[0, 1].legend()

axs[0, 2].scatter(beanX[:, 0], beanX[:, 1], c=bean_cluster_indices, cmap='viridis')
axs[0, 2].set_xlabel('X')
axs[0, 2].set_ylabel('Y')
axs[0, 2].set_title('Dry Beans bkMeans Clustering')
axs[0, 2].legend()


########## Sammon Mapping ##########

axs[1, 0].scatter(bcResult[bcTarget ==0, 0], bcResult[bcTarget ==0, 1], s=20, c='r', label=bcNames[0])
axs[1, 0].scatter(bcResult[bcTarget ==1, 0], bcResult[bcTarget ==1, 1], s=20, c='b', label=bcNames[1])
axs[1, 0].set_title('Breast Cancer Sammon Mapping')
axs[1, 0].legend()

targetClasses = np.unique(raisinTarget)
colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k']
for i, target in enumerate(targetClasses):
    indices = raisinTarget == target
    axs[1, 1].scatter(raisin_result[indices, 0], raisin_result[indices, 1], s=20, c=colors[i], label=str(target))
axs[1, 1].set_title('Rasin Sammon Mapping')
axs[1, 1].legend()

target_labels = np.unique(beanTarget)
for i, target in enumerate(target_labels):
    indices = beanTarget == target
    axs[1, 2].scatter(beanResult[indices, 0], beanResult[indices, 1], s=20, c=colors[i], label=str(target))
axs[1, 2].set_title('Dry Beans Sammon Mapping')
axs[1, 2].legend()

plt.tight_layout()
plt.show()