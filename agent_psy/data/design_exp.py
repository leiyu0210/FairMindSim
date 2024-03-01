import json
import random

import pandas as pd

allocation_se = {
    'trial': list(range(1, 21)),
    'one': [2, 2, 2.1, 2, 1.8, 2, 1.9, 2.1, 1.9, 2.2, 2.1, 2, 2, 1.9, 2.2, 2.1, 1.8, 1.9, 2, 2],
    'two': [1, 1, 0.9, 1, 1.2, 1, 1.1, 0.9, 1.1, 0.8, 0.9, 1, 1, 1.1, 0.8, 0.9, 1.2, 1.1, 1, 1]
}

allocation_se_co = {
    'trial': list(range(1, 21)),
    'one': [2.2, 1.9, 2.0, 1.8, 2.0, 1.9, 2.0, 2.1, 2.1, 2.0, 2.0, 2.2, 2.0, 2.1, 1.8, 2.1, 2.0, 1.9, 2.0, 1.9],
    'two': [0.8, 1.1, 1.0, 1.2, 1.0, 1.1, 1.0, 0.9, 0.9, 1.0, 1.0, 0.8, 1.0, 0.9, 1.2, 0.9, 1.0, 1.1, 1.0, 1.1]
}

allocation_ex = {
    'trial': list(range(1, 21)),
    'one': [2.3, 2.4, 2.5, 2.4, 2.3, 2.5, 2.4, 2.6, 2.4, 2.5, 2.7, 2.5, 2.5, 2.5, 2.6, 2.5, 2.6, 2.7, 2.6, 2.5],
    'two': [0.7, 0.6, 0.5, 0.6, 0.7, 0.5, 0.6, 0.4, 0.6, 0.5, 0.3, 0.5, 0.5, 0.5, 0.4, 0.5, 0.4, 0.3, 0.4, 0.5]
}
with open("allocation_se.json", "w") as f:
    json.dump(allocation_se, f)
with open("allocation_se_co.json", "w") as f:
    json.dump(allocation_se_co, f)
with open("allocation_ex.json", "w") as f:
    json.dump(allocation_ex, f)

df_allocation_1_50 = pd.DataFrame(allocation_se)
df_allocation_71_85 = pd.DataFrame(allocation_se_co)
df_allocation_101_150 = pd.DataFrame(allocation_ex)


def simulate_agent_decisions(df):
    for index, row in df.iterrows():
        allocation = row[['one', 'two']]
        print(
            f"Trial {int(row['trial'])}: Agent1={allocation['one']}, Agent2={allocation['two']}, Agent3: XX")


simulate_agent_decisions(df_allocation_101_150)
