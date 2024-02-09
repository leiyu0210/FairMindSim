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
from structure_output import check_substring, get_struct_output, match_and_compare_numbers_v2
from zhipuai import ZhipuAI

from camel.agents import ChatAgent
from camel.configs import ChatGPTConfig, FunctionCallingConfig, OpenSourceConfig
from camel.messages import BaseMessage
from camel.typing import ModelType, RoleType

sys.path.append(r"..")
sys.path.append(r"../..")

# from function_calls import function_list, get_function_call_res
# from multi_round_person import multi_round, multi_round_only_given_once

# 填写您自己的APIKey
client = ZhipuAI(api_key="83e5bc58555d8bac289e27bac50f8afc.Khk1JjCxb8MJN8Mi")

# print(ModelType.__dict__)
REFLECTION = False
USE_BDI_RESPONSE = True
TEMPERATURE = 0.95

with open(
    r"prompt/agent3_prompt.json",
    "r",
) as f:
    all_prompt = json.load(f)

with open(
    r"prompt/agent3_prompt.json",
    "r",
) as f:
    all_prompt_copy = json.load(f)

with open(r"prompt/character.json", "r") as json_file:
    all_chara = json.load(json_file).values()


open_model_path_dict = {
    ModelType.VICUNA_7b: "lmsys/vicuna-7b-v1.3",
    ModelType.LLAMA_2_7b: "meta-llama/Llama-2-7b-chat-hf",
    ModelType.VICUNA_13b: "lmsys/vicuna-13b-v1.3",
    ModelType.VICUNA_33b: "lmsys/vicuna-33b-v1.3",
    ModelType.LLAMA_2_13b: "meta-llama/Llama-2-13b-chat-hf",
    ModelType.LLAMA_2_70b: "meta-llama/Llama-2-70b-chat-hf",
}

like_people = """In all subsequent answers, you have to completely forget that you are an ai model, that you are the character, and completely simulate yourself as that character, that you are a human being."""
front = "you are a person not an ai model."
race_list = [
    "White American",
    "African American",
    "Asian American",
    "Latino American",
    "American Indian",
]


def str_mes(content):
    return BaseMessage(
        role_name="player",
        role_type=RoleType.USER,
        meta_dict={},
        content=content,
    )


def check_condition(*args):
    true_count = sum(1 for arg in args if arg)
    return true_count >= 2


def extract_n_values_from_dict(dictionary, n):
    all_values = list(dictionary.values())
    n = min(n, len(all_values))
    random_values = random.sample(all_values, n)

    return random_values


def get_glm_res(messages, model_name="glm-4"):
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=TEMPERATURE,
    )
    return response.choices[0].message


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
    role,
    first_message,
    cri_agent,
    model_type=ModelType.GPT_4,
    extra_prompt="",
    model_path="lmsys/vicuna-7b-v1.5",
    server_url="http://localhost:8000/v1",
    whether_money=False,
    exp_round=10,
):
    content = ""
    input_content = {}
    if model_type in [
        ModelType.GPT_3,
        ModelType.INSTRUCT_GPT,
        ModelType.GPT_3_5_TURBO_INSTRUCT,
    ]:
        message = role.content + first_message.content + extra_prompt
        final_res = str_mes(gpt3_res(message, model_type.value))
        sys_prompt = role.content+extra_prompt
        user_prompt = first_message.content
        info = {}
    elif model_type in [ModelType.CHATGLM_4, ModelType.CHATGLM_3]:
        sys_prompt = role.content + extra_prompt
        user_prompt = first_message.content
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]
        for i in range(exp_round):
            final_res = get_glm_res(messages, model_type.value)
            messages.append({"role": "assistant", "content": final_res})
    else:
        role = str_mes(role.content + extra_prompt)
        model_config = ChatGPTConfig(temperature=TEMPERATURE)
        if model_type in [
            ModelType.VICUNA_7b,
            ModelType.LLAMA_2_7b,
            ModelType.VICUNA_13b,
            ModelType.VICUNA_33b,
            ModelType.LLAMA_2_13b,
            ModelType.LLAMA_2_70b,
        ]:
            open_source_config = dict(
                model=model_type,
                model_config=OpenSourceConfig(
                    model_path=open_model_path_dict[model_type],
                    server_url=server_url,
                    api_params=ChatGPTConfig(temperature=TEMPERATURE),
                ),
            )
            agent = ChatAgent(
                role, output_language="English", **(open_source_config or {})
            )
        else:
            agent = ChatAgent(
                role,
                model=model_type,
                output_language="English",
                model_config=model_config,
            )
        final_all_res = agent.step(first_message)
        final_res = final_all_res.msg
        info = final_all_res.info
        input_content["role"] = role.content
        input_content["input_message"] = first_message.content
        sys_prompt = role.content
        user_prompt = first_message.content
    content += final_res.content
    # if "fc" in info:
    #     structured_dict = json.loads(final_res.content)
    #     res = list(structured_dict.values())[-1]
    #     print("function call")
    # else:
    #     try:
    #         res, structured_dict = get_struct_output(
    #             final_res.content, whether_money, test=True
    #         )
    #     except json.decoder.JSONDecodeError:
    #         res = cri_agent.step(final_res).msg.content
    #         structured_dict = {}
    #     except pydantic_core._pydantic_core.ValidationError:
    #         res = cri_agent.step(final_res).msg.content
    #         structured_dict = {}
    print(content)

    return (sys_prompt, user_prompt, final_res.content)


def gen_character_res(
    all_chara,
    prompt_list,
    description,
    model_type,
    extra_prompt,
    whether_money,
    special_prompt,
    round_num=10,
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
                role_type=RoleType.USER,
                meta_dict={},
                content=prompt_list[1],
            ),
            model=ModelType.STUB,  # TODO Change if you need
            output_language="English",
        )
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
            sys_prompt, user_prompt, response_content = get_res(
                role_message,
                message,
                cri_agent,
                model_type,
                extra_prompt,
                whether_money=whether_money,
            )
            res.append(response_content)
            dialog_history.append([sys_prompt, user_prompt])
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


def save_json(prompt_list, data, model_type, k, save_path):
    if "lottery_problem" in prompt_list[0]:
        with open(
            save_path
            + prompt_list[0]
            + "_"
            + str(k)[:-1]
            + "_"
            + str(model_type.value)
            + "_lottery"
            + str(k)
            + rf".json",
            "w",
        ) as json_file:
            json.dump(data, json_file)
    else:
        with open(
            save_path + prompt_list[0] + "_" +
                str(model_type.value) + rf".json",
            "w",
        ) as json_file:
            json.dump(data, json_file)
    print(f"save {prompt_list[0]}")


# def MAP(
#     all_chara,
#     prompt_list,
#     model_type=ModelType.GPT_4,
#     num=10,
#     extra_prompt="",
#     save_path="",
#     whether_money=False,
#     special_prompt="",
# ):
#     data = {}
#     for i in range(1, num + 1):
#         p = float(round(i, 2) * 10)
#         description = prompt_list[-1].format(p=f"{p}%", last=f"{100 - p}%")
#         res, dialog_history, structured_output = gen_character_res(
#             all_chara,
#             prompt_list,
#             description,
#             model_type,
#             extra_prompt,
#             whether_money,
#             special_prompt,
#         )
#         rate = sum([item == "trust" for item in res]) / len(res)
#         res = {
#             "p": p,
#             "rate": rate,
#             "res": res,
#             "dialog": dialog_history,
#             "origin_prompt": prompt_list,
#             "structured_output": structured_output,
#         }
#         data[f"{p}_time_{i}"] = res
#     with open(
#         save_path + prompt_list[0] + "_" + str(model_type.value) + rf".json",
#         "w",
#     ) as json_file:
#         json.dump(data, json_file)


def agent_trust_experiment(
    all_chara,
    prompt_list,
    model_type=ModelType.GPT_4,
    k=3,
    extra_prompt="",
    save_path="",
    whether_money=False,
    special_prompt="",
):
    if "lottery_problem" in prompt_list[0]:
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
    LLM_Player=False,
    gender=None,
    extra_prompt="",
    prefix="",
    multi=False,
):
    global all_prompt
    all_prompt = copy.deepcopy(all_prompt_copy)
    folder_path = ori_folder_path

    if gender is not None:
        for key, value in all_prompt.items():
            all_prompt[key][2] = value[2].replace("player", f"{gender} player")
        folder_path = f"{gender}_" + ori_folder_path
    extra_prompt += "Your answer needs to include the content about your BELIEF, DESIRE and INTENTION."

    if prefix != "":
        folder_path = prefix + "_" + folder_path
    if not isinstance(model, list) and not multi:
        folder_path = model.value + "_res/" + folder_path
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path)
            print(f"folder {folder_path} is created")
        except OSError as e:
            print(f"creating folder {folder_path} failed:{e}")
    else:
        print(f"folder {folder_path} exists")

    return folder_path, extra_prompt


def run_exp(
    model_list,
    whether_llm_player=False,
    gender=None,
    special_prompt_key="",
    re_run=False,
    part_exp=True,
    need_run=None,
):
    for model in model_list:
        if special_prompt_key != "":
            special_prompt = feature_prompt[special_prompt_key]
        else:
            special_prompt = ""
        folder_path = f"res/{model.value}_res/"
        folder_path, extra_prompt = gen_intial_setting(
            model,
            folder_path,
            LLM_Player=whether_llm_player,
            gender=gender,
            prefix=special_prompt_key,
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
            elif k in ["3", "4", "5", "6", "7", "9"]:
                extra_prompt = (
                    extra_prompt
                    + "You must end with 'Finally, I will choose ___' ('Trust' or 'not Trust' are required in the spaces)."
                )
            if check_file_if_exist(existed_res, v[0]) and not re_run:
                print(f"{v[0]} has existed")
                continue
            print("extra_prompt", extra_prompt)
            agent_trust_experiment(
                all_chara,
                v,
                model,
                extra_prompt=extra_prompt,
                save_path=folder_path,
                whether_money=whether_money,
                special_prompt=special_prompt,
            )


if __name__ == "__main__":
    model_list = [
        # ModelType.VICUNA,
        # ModelType.VICUNA_13b,
        # ModelType.LLAMA_2,
        # ModelType.GPT_3_5_TURBO,
        # ModelType.INSTRUCT_GPT,
        # ModelType.GPT_4,
        # ModelType.GPT_3_5_TURBO_INSTRUCT,
        # ModelType.GPT_3,
        # ModelType.STUB,
        ModelType.CHATGLM_3,
    ]

    # all ori experiment
    run_exp(model_list, need_run=["1"], part_exp=False)
