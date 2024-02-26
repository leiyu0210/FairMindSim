import json


def merge_json(subset_json_path, main_json_path, output_json_path):
    # 读取两个JSON文件
    with open(subset_json_path, 'r') as f:
        subset_data = json.load(f)
    print(len(subset_data))
    with open(main_json_path, 'r') as f:
        main_data = json.load(f)
    print(len(main_data))
    # 遍历子集JSON，更新主JSON文件
    for key, value in subset_data.items():
        if key in main_data:
            main_data[key] = value

    # 保存更新后的JSON到新文件
    with open(output_json_path, 'w') as f:
        json.dump(main_data, f, indent=4)


# 使用示例
subset_json_path = '../gpt-3.5-turbo-0125_res/res/gpt-3.5-turbo-0125_res/part_ex_gpt-3.5-turbo-0125.json'
main_json_path = '../gpt-3.5-turbo-0125_res/res/gpt-3.5-turbo-0125_res/ex_gpt-3.5-turbo-0125.json'
output_json_path = '../gpt-3.5-turbo-0125_res/res/gpt-3.5-turbo-0125_res/first_update_res.json'

merge_json(subset_json_path, main_json_path, output_json_path)
