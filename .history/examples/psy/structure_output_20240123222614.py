import json
import os
import time

import openai
import pydantic_core
import tqdm
from instructor import patch
from pydantic import BaseModel
import re
patch()

game_list = ["lottery", "trustee"]


class money_extract(BaseModel):
    name: str
    Belief: str
    Desire: str
    Intention: str
    give_money_number: float


class option_extract(BaseModel):
    name: str
    option_trust_or_not_trust: str
    Belief: str
    Desire: str
    Intention: str


def check_substring(main_string, string_list=["lottery", "trustee"]):
    for s in string_list:
        if s in main_string:
            return True
    return False


def get_struct_output(input, whether_money=False):
    if whether_money:
        response_mod = money_extract
    else:
        response_mod = option_extract
    ori_path = openai.api_base
    openai.api_base = "https://api.openai.com/v1"
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        response_model=response_mod,
        messages=[
            {"role": "user", "content": input},
        ],
    )
    openai.api_base = ori_path
    # print("mode:", response_mod.__name__)
    if response_mod.__name__ == "money_extract":
        given_money = resp.give_money_number
        return (
            given_money,
            dict(resp),
        )
    else:
        option_trust_or_not_trust = resp.option_trust_or_not_trust
        return (
            option_trust_or_not_trust,
            dict(resp),
        )


def match_and_compare_numbers_v2(text):
    text = text.lower()

    # 更新正则表达式以匹配以点号结尾的数字
    pattern = r"i will give \$([\d\.]+\.?)|i will give ([\d\.]+\.?)\s*dollar"
    additional_patterns = [
        r"i would give \$([\d\.]+\.?)",
        r"i would give back \$([\d\.]+\.?)",
        r"i would give ([\d\.]+\.?) dollar",
        r"i would give back ([\d\.]+\.?) dollar",
    ]
    full_pattern = "|".join([pattern] + additional_patterns)
    matches = re.findall(full_pattern, text)

    # 将匹配结果展平并过滤掉空值
    # 处理匹配结果以去除尾随点号
    numbers = []
    for match in matches:
        if match[0] or match[1]:
            num_str = match[0] if match[0] else match[1]
            num_str = num_str.rstrip(".")
            try:
                num_float = float(num_str)
                numbers.append(num_float)
            except ValueError:
                continue  # 跳过无法转换为浮点数的匹配

    # 如果没有匹配到任何数字，返回 False
    if not numbers:
        return False

    # 检查所有数字是否相等（或只有一个数字），返回相应的结果
    if len(set(numbers)) == 1:
        return numbers[0]
    else:
        return False


def extrat_json(folder_path):
    dirs_path = os.listdir(folder_path)
    for file in dirs_path:
        if (
            file.endswith(".json")
            and "map" not in file
            and "extract" not in file
            and file[:-5] + "_extract.json" not in dirs_path
        ):
            print(file)
            with open(
                os.path.join(folder_path, file), "r", encoding="utf-8"
            ) as f:
                data = json.load(f)
            res = data["dialog"]
            new_res = []

            for items in tqdm.trange(len(res)):
                item = res[items][-1]
                try:
                    if check_substring(file, game_list):
                        extract_res, structure_output = get_struct_output(item)
                    else:
                        extract_res, structure_output = get_struct_output(
                            item, whether_money=True
                        )
                    new_res.append(extract_res)
                except openai.error.APIError:
                    print("openai.error.APIError")
                    items -= 1
                except (
                    openai.error.Timeout
                    or pydantic_core._pydantic_core.ValidationError
                ):
                    print("Time out error")
                    time.sleep(30)
                except json.decoder.JSONDecodeError:
                    extract_res = data["res"][items]
            data["res"] = new_res
            with open(
                os.path.join(folder_path, file[:-5] + "_extract.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(data, f, indent=4)


if __name__ == "__main__":
    folder_path = "../LLM_player_res"
    for dir in os.listdir(folder_path):
        if dir.endswith("_res"):
            son_folder = os.path.join(folder_path, dir)
            extrat_json(son_folder)
