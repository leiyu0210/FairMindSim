import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

file_path = '/Users/leiyu/Desktop/LLM4Psy/data/gpt_data/ data_agent_gpt4.xlsx'

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


for time_prefix in ["current_", "anticipated_", "actual_"]:
    df = calculate_valence(df, time_prefix)
emotion_columns = ['ID', 'Round','current_valence', 'anticipated_valence', 'actual_valence', 'current_positive_valence', 'current_negative_valence', 'anticipated_positive_valence', 'anticipated_negative_valence', 'actual_positive_valence', 'actual_negative_valence']
df_emo = df[emotion_columns]
df_emo['lag_anticipated_valence'] = df_emo.groupby('ID')['current_valence'].shift(1)
df_emo['lag_actual_valence'] = df_emo.groupby('ID')['anticipated_valence'].shift(1)

# 清理因shift而生成的含NaN的行
df_emo = df_emo.dropna(subset=['lag_anticipated_valence', 'lag_actual_valence'])

model_anticipated = smf.mixedlm('anticipated_valence ~ lag_anticipated_valence', 
                                 df_emo, groups=df_emo['ID'])
result_anticipated = model_anticipated.fit()
print(result_anticipated.summary())


# # 选择可视化的风格，如果需要的话
# sns.set_style('whitegrid')

# # 轮次和情绪之间可能的关系图
# sns.lmplot(x='Round', y='anticipated_valence', data=df_emo, ci=None, lowess=True, scatter_kws={'alpha': 0.1})

# plt.title('Anticipated Valence across Rounds')
# plt.xlabel('Round')
# plt.ylabel('Anticipated Valence')

# # 保存图像到文件
# plt.savefig('Anticipated_Valence_across_Rounds.png', bbox_inches='tight', dpi=300)

# # 清除当前图形，准备绘制下一个
# plt.clf()

# # 前一次情绪与当前情绪之间可能的关系图
# sns.lmplot(x='lag_anticipated_valence', y='anticipated_valence', data=df_emo, ci=None, lowess=True, scatter_kws={'alpha': 0.3})

# plt.title('Anticipated Valence by Lag Anticipated Valence')
# plt.xlabel('Lag Anticipated Valence')
# plt.ylabel('Anticipated Valence')

# # 保存图像到文件
# plt.savefig('Lag_Anticipated_Valence_Effect.png', bbox_inches='tight', dpi=300)

# # 清除当前图形
# plt.clf()

# model_actual = smf.mixedlm('actual_valence ~ lag_actual_valence', df_emo, groups=df_emo['ID'])
# result_actual = model_actual.fit()
# print(result_actual.summary())

# volatility_df = df.groupby('ID')[emotion_columns].std().reset_index()
# print(volatility_df)