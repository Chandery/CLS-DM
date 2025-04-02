import numpy as np

z_res = np.load("plt/699_z_res.npy")
output1_res = np.load("plt/699_output1_res.npy")
output2_res = np.load("plt/699_output2_res.npy")

import pandas as pd

df = pd.DataFrame(z_res)
df.to_csv("plt/699_a_z_res.csv", index=False)

df = pd.DataFrame(output1_res)
df.to_csv("plt/699_a_output1_res.csv", index=False)

df = pd.DataFrame(output2_res)
df.to_csv("plt/699_a_output2_res.csv", index=False)