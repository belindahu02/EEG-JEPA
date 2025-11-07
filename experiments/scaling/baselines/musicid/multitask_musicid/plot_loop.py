from plot_results import *

# single layer: ft=5
# 2 layer: ft=8
# 3 layer: ft=11
# 4 layer: ft=12
# all layer: ft=17

#0 #17
# single layer: ft=5 #12
# 2 layer: ft=6 #11
# 3 layer: ft=9 #8
# 4 layer: ft=12 #5
# all layer: ft=17 #0

# For scaling experiment, test with 20 classes (all users)
num_classes = 20

for layers in [0,1,2,3,4,5]:
  plotspu(layers, num_classes)
