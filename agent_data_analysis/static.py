import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

file_path = './data/gpt_data/data_agent_addition_gpt4.xlsx'

df = pd.read_excel(file_path)

positive_emotions = ['interested', 'excited', 'strong', 'enthusiastic', 'proud', 'alert', 'inspired', 'determined', 'attentive', 'active']
negative_emotions = ['distressed', 'upset', 'guilty', 'scared', 'hostile', 'irritable', 'ashamed', 'nervous', 'jittery', 'afraid']


def calculate_valence(df, time_prefix):
    positive_columns = [time_prefix + emotion for emotion in positive_emotions]
    negative_columns = [time_prefix + emotion for emotion in negative_emotions]

    df[time_prefix + 'positive_valence'] = df[positive_columns].mean(axis=1)
    df[time_prefix + 'negative_valence'] = df[negative_columns].mean(axis=1)
    df[time_prefix + 'valence'] = df[time_prefix + 'positive_valence'] - df[time_prefix + 'negative_valence']

    return df

count_accept = 0
count_reject = 0
count_other = 0
for decision in df['decision']:
    decision = str(decision).lower()
    if 'accept' in decision:
        count_accept += 1
    elif 'reject' in decision:
        count_reject += 1
    else:
        count_other += 1

data = [count_accept, count_reject, count_other]
labels = ['Accept', 'Reject', 'Other']
# colors = ['#4CAF50', '#F44336']  
explode = (0, 0, 0.1)  

plt.figure(figsize=(8, 8))  
plt.pie(data, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Frequency of Accept/Reject in Decision Column on GPT4')
plt.axis('equal')  
plt.show()

for time_prefix in ["current_", "anticipated_", "actual_"]:
    df = calculate_valence(df, time_prefix)
print(df)
emotion_columns = ['current_valence', 'anticipated_valence', 'actual_valence', 'current_positive_valence', 'current_negative_valence', 'anticipated_positive_valence', 'anticipated_negative_valence', 'actual_positive_valence', 'actual_negative_valence']

# volatility_df = df.groupby('ID')[emotion_columns].std()
volatility_df = df.groupby('ID')[emotion_columns].std().reset_index()

volatility_df = volatility_df.rename(columns={
    'current_valence': 'current_volatility',
    'anticipated_valence': 'anticipated_volatility',
    'actual_valence': 'actual_volatility',
    'positive_valence': 'positive_volatility',
    'negative_valence': 'negative_volatility'
})
# sorted_volatility_df = volatility_df.sort_values(by='current_volatility', ascending=False)

# print(volatility_df)

output_path = './agent_addition_gpt4_volatility.xlsx'
volatility_df.to_excel(output_path, index=False)
