import json
import pandas as pd

def extract_panas_data_to_excel(json_data):
    processed_data = []

    for individual_id, rounds in json_data.items():
        for round_id, content_list in rounds.items():
            json_string = content_list[1]
            try:
                round_data = json.loads(json_string)
                round_number = round_id.split('_')[1]
                round_data['ID'] = individual_id
                round_data['Round'] = int(round_number)  
                processed_data.append(round_data)
            except json.JSONDecodeError:
                print(f"An error occurred while decoding JSON for individual {individual_id}, {round_id}")

    df = pd.DataFrame(processed_data)

    column_order = ['ID', 'Round'] + [col for col in df.columns if col not in ['ID', 'Round']]
    df = df[column_order]
#     df.to_excel(excel_file_path, index=False)

json_data = data
extract_panas_data_to_excel(json_data)
