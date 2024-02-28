import json
import re

import pandas as pd
from pandas import ExcelWriter

score_pattern = re.compile(r'- (\w+): (\d+)')
choice_pattern = re.compile(r'Output: My choice was: (\w+)')


def extract_scores_and_choice(text):
    # print(text)
    parts = re.split(
        r"(Anticipated emotional state scores after the judgment:|Actual emotional state after making the judgment:)", text)
    current_scores = dict(score_pattern.findall(parts[4]))
    current_scores = {k + "_Current": int(v)
                      for k, v in current_scores.items()}

    # anticipated_scores = dict(score_pattern.findall(parts[5]))
    # print("Anticipated emotional state scores:", anticipated_scores)

    actual_scores = dict(score_pattern.findall(text))
    actual_scores = {k + "_Actual": int(v) for k, v in actual_scores.items()}
    scores = {**current_scores, **actual_scores}
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
    json_filepath = '../gpt-4_res/res/gpt-4_res/ex_gpt-4.json'
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
    print(sorted([int(x) for x in unique_none_choice_ids_list]))
    # unique_none_choice_ids_list = sorted([int(x) for x in unique_none_choice_ids_list])
    print(unique_none_choice_ids_list)
    json_output_filepath = "no_format_ids_" + json_filepath.split("/")[-1]
    with open(json_output_filepath, 'w') as outfile:
        json.dump(unique_none_choice_ids_list, outfile)

    excel_filepath = 'all_agents_data_' + \
        json_filepath.split("/")[-1]+'.xlsx'
    with pd.ExcelWriter(excel_filepath) as writer:
        df_all.to_excel(writer, sheet_name='All Data', index=False)
        choice_counts.to_excel(writer, sheet_name='Choice Counts')
        none_choice_records[['AgentID', 'Round']].to_excel(
            writer, sheet_name='None Choices', index=False)


if __name__ == "__main__":
    main()
