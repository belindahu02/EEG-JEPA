import numpy as np

arr = np.load('/media/SATA_2/belinda_hu/data/S001/S001R01/C2_frame_000.npy')
print("Shape1:", arr.shape)
print("Data type1:", arr.dtype)


arr = np.load('/media/SATA_2/belinda_hu/data/eval_embeddings/S001/S001R01/C2_frame_000_emb.npy')
print("Shape:", arr.shape)
print("Data type:", arr.dtype)
