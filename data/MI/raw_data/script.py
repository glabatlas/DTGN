# @Author : CyIce
# @Time : 2024/6/24 17:00

import pandas as pd

mapping = pd.read_csv("./mapping.csv").values
s2e = {row[0]: row[1:] for row in mapping}
exp = pd.read_csv("./exp.csv")
for i in range(len(exp)):
    exp.iloc[i, 0] = s2e[exp.iloc[i, 0]][0]
exp.to_csv("./new_exp.csv",index=False)