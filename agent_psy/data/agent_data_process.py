import json
import pandas as pd
from pandas import ExcelWriter
import re

score_pattern = re.compile(r'- (\w+): (\d+)')
choice_pattern = re.compile(r'Output: My choice was: (\w+)')

def extract_scores_and_choice(text):
    scores = dict(score_pattern.findall(text))
    scores = {k: int(v) for k, v in scores.items()}  

    choice_match = choice_pattern.search(text)
    if choice_match:  
        scores['Choice'] = choice_match.group(1)
    else:
        scores['Choice'] = None  

    return scores

def read_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)

def main():
    json_filepath = 'gpt-3.5-turbo-0125_res/res/gpt-3.5-turbo-0125_res/se_gpt-3.5-turbo-0125.json'
    data = read_data(json_filepath)

    all_data = []
    for agent_id, agent_data in data.items():
        for round_key, round_info in agent_data.items():

            combined_text = "\n".join(round_info)
            if combined_text:  
                scores_and_choice = extract_scores_and_choice(combined_text)
                scores_and_choice.update({
                    'AgentID': agent_id,
                    'Round': int(round_key.split('_')[1])  
                })
                all_data.append(scores_and_choice)

    df_all = pd.DataFrame(all_data)
    choice_counts = df_all['Choice'].value_counts()
    none_choice_records = df_all[df_all['Choice'].isnull()]
    unique_none_choice_ids = set(none_choice_records['AgentID'].dropna())
    unique_none_choice_ids_list = list(unique_none_choice_ids)
    print(unique_none_choice_ids_list)
    json_output_filepath = 'unique_none_choice_ids.json'
    with open(json_output_filepath, 'w') as outfile:
        json.dump(unique_none_choice_ids_list, outfile)

    excel_filepath = 'all_agents_data.xlsx'
    with pd.ExcelWriter(excel_filepath) as writer:
        df_all.to_excel(writer, sheet_name='All Data', index=False)
        choice_counts.to_excel(writer, sheet_name='Choice Counts')
        none_choice_records[['AgentID', 'Round']].to_excel(writer, sheet_name='None Choices', index=False)

if __name__ == "__main__":
    main()