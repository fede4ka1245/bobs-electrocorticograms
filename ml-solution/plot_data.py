import numpy as np
from matplotlib import pyplot as plt

from utils import create_dataset

file_paths = [
    'nmic_ivnd_dataset/ECoG_fully_marked_(4+2 files, 6 h each)/Ati4x1_15m_BL_6h_fully_marked.edf',
    'nmic_ivnd_dataset/ECoG_fully_marked_(4+2 files, 6 h each)/Ati4x1_15m_Dex003(Pharm!)_6h_fully_marked.edf',
    'nmic_ivnd_dataset/ECoG_fully_marked_(4+2 files, 6 h each)/Ati4x1_15m_H2O_6h_fully_marked.edf',
    'nmic_ivnd_dataset/ECoG_fully_marked_(4+2 files, 6 h each)/Ati4x3_9m_Xyl01(Pharm!)_6h_fully_marked.edf',
    'nmic_ivnd_dataset/ECoG_fully_marked_(4+2 files, 6 h each)/Ati4x3_12m_BL_6h_fully_marked.edf',
    'nmic_ivnd_dataset/ECoG_fully_marked_(4+2 files, 6 h each)/Ati4x6_14m_BL_6h_fully_marked.edf'
]

segments, labels = create_dataset(file_paths)

unique_labels = [0, 1, 2, 3]
print(unique_labels)

fig, axes = plt.subplots(4, 1, figsize=(10, 6))

for i, label in enumerate(unique_labels):
    idx = np.where(labels == label)[0][10]

    axes[i].plot(segments[idx][0, :], label=f'channel 1, label={label}')
    axes[i].plot(segments[idx][1, :], label=f'channel 2, label={label}')
    # print(np.where(labels == label)[0][10 - 5:10 + 5])
    axes[i].plot(segments[idx, 2, :].reshape(-1), label=f'channel 3, label={label}')
    axes[i].legend()
plt.legend()
plt.show()
