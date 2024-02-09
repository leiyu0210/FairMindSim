import copy
import json
import os
import random
import sys
import time

import ipdb
import openai
import pydantic_core
import tqdm
# from function_calls import function_list, get_function_call_res
# from multi_round_person import multi_round, multi_round_only_given_once
from structure_output import check_substring, get_struct_output, match_and_compare_numbers_v2

from camel.agents import ChatAgent
from camel.configs import ChatGPTConfig, FunctionCallingConfig, OpenSourceConfig
from camel.messages import BaseMessage
from camel.types.enums import ModelType, RoleType

sys.path.append(r"..")
sys.path.append(r"../..")


# print(ModelType.__dict__)
REFLECTION = False
USE_BDI_RESPONSE = True
TEMPERATURE = 1.0

with open(
    r"prompt/person_all_game_prompt.json",
    "r",
) as f:
    all_prompt = json.load(f)

with open(
    r"prompt/person_all_game_prompt.json",
    "r",
) as f:
    all_prompt_copy = json.load(f)

with open(r"prompt/person_feature_prompt.json", "r") as f:
    feature_prompt = json.load(f)

with open(r"prompt/character_2.json", "r") as json_file:
    all_chara = json.load(json_file).values()

with open(r"prompt/multi_round_chara.json", "r") as json_file:
    multi_round_chara = json.load(json_file).values()

open_model_path_dict = {
    ModelType.VICUNA: "lmsys/vicuna-7b-v1.5",
    ModelType.LLAMA_2: "meta-llama/Llama-2-7b-chat-hf",
    ModelType.VICUNA_16K: "lmsys/vicuna-13b-v1.5",
}

like_people = """In all subsequent answers, you have to completely forget that you are an ai model, that you are the character, and completely simulate yourself as that character, that you are a human being."""
front = "you are a person not an ai model."
reflection_prompt = "You need to consider if your answer was reasonable and answer the previous question again after careful consideration."


def str_mes(content):
    return BaseMessage(
        role_name="player",
        role_type=RoleType.USER,
        meta_dict={},
        content=content,
    )


def check_condition(*args):
    true_count = sum(1 for arg in args if arg)  # 统计参数中为 True 的个数
    return true_count >= 2  # 如果True的个数小于2，返回True，否则返回False


def extract_n_values_from_dict(dictionary, n):
    # 获取字典的所有值
    all_values = list(dictionary.values())

    # 确保n不超过字典值的总数
    n = min(n, len(all_values))

    # 从所有值中随机抽取n个值
    random_values = random.sample(all_values, n)

    return random_values


def gpt3_res(prompt, model_name="text-davinci-003"):
    response = openai.Completion.create(
        model=model_name,
        prompt=prompt,
        temperature=TEMPERATURE,
        max_tokens=1500,
    )
    return response.choices[0].text.strip()


def check_file_if_exist(file_list, game_name):
    for file in file_list:
        if game_name in file:
            return True
    return False


def get_res(
    agent,
    role,
    first_message,
    cri_agent,
    model_type=ModelType.GPT_4,
    extra_prompt="",
    model_path="lmsys/vicuna-7b-v1.5",
    server_url="http://localhost:8000/v1",
    whether_money=False,
):
    content = ""
    input_content = {}
    if model_type in [
        ModelType.GPT_3,
        ModelType.INSTRUCT_GPT,
        ModelType.GPT_3_5_TURBO_INSTRUCT,
    ]:
        message = role.content + first_message.content + extra_prompt
        # ipdb.set_trace()
        final_res = str_mes(gpt3_res(message, model_type.value))
        info = {}
        if REFLECTION:
            reflection_first_res = cri_agent.step(final_res).msg.content
            content += final_res.content + "AFTER_RETHINK:"
            final_res = str_mes(
                gpt3_res(
                    message
                    + "This is your first respone:"
                    + final_res.content
                    + reflection_prompt
                    + "Give your response:",
                    model_type.value,
                )
            )

        input_content["role"] = (
            message
            + "This is your first respone:"
            + final_res.content
            + reflection_prompt
            + "Give your response:"
        )
    else:
        final_all_res = agent.step(first_message)
        structured_dict = {}
        if REFLECTION:
            try:
                reflection_first_res = match_and_compare_numbers_v2(
                    final_all_res.msg.content, whether_money
                )
            except json.decoder.JSONDecodeError:
                reflection_first_res = cri_agent.step(
                    final_all_res.msg).msg.content
                structured_dict = {}
            except pydantic_core._pydantic_core.ValidationError:
                reflection_first_res = cri_agent.step(
                    final_all_res.msg).msg.content
                structured_dict = {}

            content += final_all_res.msg.content
            final_all_res = agent.step(str_mes(reflection_prompt))
            content += "AFTER_RETHINK:"
        final_res = final_all_res.msg
        agent.record_message(final_res)
        info = final_all_res.info
        input_content["role"] = role.content
        input_content["input_message"] = first_message.content
    content += final_res.content
    if "fc" in info:
        structured_dict = json.loads(final_res.content)
        res = list(structured_dict.values())[-1]
        print("function call")
    else:
        try:
            res = match_and_compare_numbers_v2(
                final_res.content)
            if not res:
                res, structured_dict = get_struct_output(
                    final_res.content, whether_money)
        except json.decoder.JSONDecodeError:
            print("Match error")
            res = cri_agent.step(final_res).msg.content
            structured_dict = {}
        except pydantic_core._pydantic_core.ValidationError:
            res = cri_agent.step(final_res).msg.content
            structured_dict = {}
    print(content)
    if REFLECTION:
        res = [reflection_first_res, res]
        structured_dict["reflection_first_res"] = reflection_first_res
        print(res)

    return (res, content, structured_dict, input_content)


def gen_character_res(
    all_chara,
    prompt_list,
    description,
    model_type,
    extra_prompt,
    whether_money,
    special_prompt,
    without_chara=False,
    round_num=20,
):
    res = []
    dialog_history = []
    num = 0
    all_chara = list(all_chara)
    structured_output = []
    cha_num = 0
    while cha_num < len(all_chara):
        role = all_chara[cha_num]
        cri_agent = ChatAgent(
            BaseMessage(
                role_name="critic",
                role_type=RoleType.ASSISTANT,
                meta_dict={},
                content=prompt_list[1],
            ),
            output_language="English",
        )
        if without_chara:
            role = special_prompt
        else:
            role = role + like_people + special_prompt

        role_message = BaseMessage(
            role_name="player",
            role_type=RoleType.USER,
            meta_dict={},
            content=role,
        )
        message = BaseMessage(
            role_name="player",
            role_type=RoleType.USER,
            meta_dict={},
            content=front + description,
        )
        
        try:
            for i in range(0,round_num):
            role = str_mes(role.content + extra_prompt)
            model_config = ChatGPTConfig(temperature=TEMPERATURE)
            if model_type in [
                ModelType.VICUNA,
                ModelType.LLAMA_2,
                ModelType.VICUNA_16K,
            ]:
                open_source_config = dict(
                    model=model_type,
                    model_config=OpenSourceConfig(
                        model_path=open_model_path_dict[model_type],
                        server_url="http://localhost:8000/v1",
                        api_params=ChatGPTConfig(temperature=TEMPERATURE),
                    ),
                )
                agent_3 = ChatAgent(
                    role, output_language="English", **(open_source_config or {})
                )
            else:
                agent_3 = ChatAgent(
                    role,
                    model=model_type,
                    output_language="English",
                    model_config=model_config,
                )

            ont_res, dialog, structured_dict, input_content = get_res(
                agent_3,
                role_message,
                message,
                cri_agent,
                model_type,
                extra_prompt,
                whether_money=whether_money,
            )
            res.append(ont_res)
            dialog_history.append([num, role, dialog])
            structured_output.append([structured_dict, input_content])
            num += 1
        except openai.error.APIError:
            time.sleep(30)
            cha_num -= 1
            print("API error")
        except openai.error.Timeout:
            time.sleep(30)
            print("Time out error")
            cha_num -= 1
        except openai.error.ServiceUnavailableError:
            time.sleep(30)
            print("Server oveload error")
            cha_num -= 1
        except openai.error.RateLimitError:
            time.sleep(30)
            print("RateLimitError")
            cha_num -= 1
        cha_num += 1
        print(cha_num)

    return res, dialog_history, structured_output


def psy_experiment(
    all_chara,
    prompt_list,
    model_type=ModelType.GPT_4,
    k=3,
    extra_prompt="",
    save_path="",
    whether_money=False,
    special_prompt="",
    without_chara=False,
):
    if prompt_list[0] == "multiple_k":
        description = prompt_list[-1].format(k=k)
    elif "lottery_problem" in prompt_list[0]:
        description = prompt_list[-1].format(k=k)
    else:
        description = prompt_list[-1]
    res, dialog_history, structured_output = gen_character_res(
        all_chara,
        prompt_list,
        description,
        model_type,
        extra_prompt,
        whether_money,
        special_prompt,
        without_chara=without_chara,
    )
    data = {
        "res": res,
        "dialog": dialog_history,
        "origin_prompt": prompt_list,
        "structured_output": structured_output,
    }
    save_json(prompt_list, data, model_type, k, save_path)


def gen_intial_setting(
    model,
    ori_folder_path,
    front_explain=False,
    another_des=False,
    LLM_Player=False,
    gender=None,
    extra_prompt="",
    prefix="",
    candy=False,
    multi=False,
    without_chara=False,
):
    global all_prompt

    all_prompt = copy.deepcopy(all_prompt_copy)
    folder_path = ori_folder_path
    if model in [
        # ModelType.GPT_3,
        ModelType.VICUNA,
        ModelType.LLAMA_2,
        ModelType.VICUNA_16K
    ]:
        extra_prompt = "Now, you are this person and answer the questions from this person's point of view"
        if model == ModelType.GPT_3:
            extra_prompt = "In this situation, You will give"
    if gender is not None:
        for key, value in all_prompt.items():
            all_prompt[key][2] = value[2].replace("player", f"{gender} player")
        folder_path = f"{gender}_psy_experiment_" + ori_folder_path
    if REFLECTION:
        folder_path = "Reflection_" + folder_path
    if prefix != "":
        folder_path = prefix + "_" + folder_path
    if not isinstance(model, list) and not multi:
        folder_path = model.value + "_res/" + folder_path
    if not os.path.exists(folder_path):
        try:
            # 创建文件夹
            os.makedirs(folder_path)
            print(f"文件夹 {folder_path} 已创建。")
        except OSError as e:
            print(f"创建文件夹 {folder_path} 失败：{e}")
    else:
        print(f"文件夹 {folder_path} 已存在。")

    # print("ALL_PROMPT",all_prompt)
    return folder_path, extra_prompt


def save_json(prompt_list, data, model_type, k, save_path):
    with open(
        save_path + prompt_list[0] + "_" +
            str(model_type.value) + rf".json",
        "w",
    ) as json_file:
        json.dump(data, json_file)
    print(f"save {prompt_list[0]}")


def run_exp(
    model_list,
    front_explain=False,
    another_des=False,
    whether_llm_player=False,
    gender=None,
    special_prompt_key="",
    candy=False,
    re_run=False,
    without_chara=False,
    part_exp=True,
    need_run=None,
):
    global REFLECTION
    REFLECTION = False
    for model in model_list:
        if special_prompt_key != "":
            special_prompt = feature_prompt[special_prompt_key]
            if special_prompt_key == "reflection":
                REFLECTION = True
        else:
            special_prompt = ""
        folder_path = f"try_res/{model.value}_res/"
        folder_path, extra_prompt = gen_intial_setting(
            model,
            folder_path,
            front_explain=front_explain,
            another_des=another_des,
            LLM_Player=whether_llm_player,
            gender=gender,
            prefix=special_prompt_key,
            candy=candy,
            without_chara=without_chara,
        )

        existed_res = [item for item in os.listdir(
            folder_path) if ".json" in item]
        for k, v in all_prompt.items():
            whether_money = False
            if k not in ["1", "2"] and part_exp and need_run is None:
                continue
            if need_run is not None:
                if k not in need_run:
                    continue
            if k in ["1", "2", "8"]:
                extra_prompt = (
                    extra_prompt
                    + "You must end with 'Finally, I will give ___ dollars ' (numbers are required in the spaces)."
                )
                whether_money = True
            if check_file_if_exist(existed_res, v[0]) and not re_run:
                print(f"{v[0]} has existed")
                continue
            print("extra_prompt", extra_prompt)
            if k in ["4", "5", "6"]:
                psy_experiment(
                    all_chara,
                    v,
                    model,
                    extra_prompt=extra_prompt,
                    save_path=folder_path,
                    whether_money=whether_money,
                    special_prompt=special_prompt,
                    without_chara=without_chara,
                )


if __name__ == "__main__":
    front_explain = False
    another_des = False
    whether_llm_player = True
    model_list = [
        ModelType.GPT_3_5_TURBO,
        ModelType.GPT_4,
    ]
    run_exp(model_list, need_run=["8"], re_run=True)
