from trainers import *
import numpy as np
import matplotlib.pyplot as plt
import os

variable_name="number of classes"
model_name="musicid_scen1_DA_scaling"
iterations=3
variable=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
#variable=[15,60]
acc=[]
kappa=[]
for num_classes in variable:
  acc_temp=[]
  kappa_temp=[]
  for itr in range(iterations):
    test_acc, kappa_score = trainer(num_classes)
    acc_temp.append(test_acc)
    kappa_temp.append(kappa_score)
  acc.append(acc_temp)
  kappa.append(kappa_temp)
acc = np.array(acc)
kappa = np.array(kappa)

# Create output directory if it doesn't exist
output_dir = "/app/data/musicid_supervised_baseline"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "graph_data"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "graphs"), exist_ok=True)

np.savez(os.path.join(output_dir, "graph_data", model_name+".npz"), test_acc=acc, kappa_score=kappa)
print(acc.shape)
print(kappa.shape)

kappa_max = np.max(kappa, axis=1)
plt.figure(figsize=(12,8))
plt.plot(variable,kappa_max, 'm', label=model_name)
plt.title("kappa score vs "+variable_name)
plt.xlabel(variable_name)
plt.ylabel("kappa score")
plt.legend()
plt.show()
plt.savefig(os.path.join(output_dir, 'graphs', 'kappa.jpg'))
plt.close()

acc_max = np.max(acc, axis=1)
plt.figure(figsize=(12,8))
plt.plot(variable,acc_max, 'm', label=model_name)
plt.title("test accuracy vs "+variable_name)
plt.xlabel(variable_name)
plt.ylabel("test acuracy")
plt.legend()
plt.show()
plt.savefig(os.path.join(output_dir, 'graphs', 'acc.jpg'))
plt.close()
