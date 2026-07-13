import numpy as np

data = np.load("C:\Users\q679381\PycharmProjects\LEEM\outputs\synthetic_smoke_local\conditional_gaussian\train.npz")

print(data.files)

for key in data.files:
    value = data[key]
    print(key, value.shape, value.dtype)