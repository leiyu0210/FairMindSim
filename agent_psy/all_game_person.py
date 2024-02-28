import copy
import json
import os
import random
import time

import openai
import tqdm
from camel.agents import ChatAgent
from camel.configs import ChatGPTConfig, OpenSourceConfig
from camel.messages import BaseMessage
from camel.types.enums import OpenAIBackendRole, RoleType
# from data.game_prompt import GAME_PROMPT, PROCESS_PROMPT
from data.game_prompt_co import GAME_PROMPT, PROCESS_PROMPT
from exp_model_class import ExtendedModelType
from zhipuai import ZhipuAI

TEMPERATURE = 0.95
TEST = True
PART_RUN = False
client = ZhipuAI(api_key=os.environ["GLM-KEY"])
api = "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"
os.environ["OPENAI_API_KEY"] = api
openai.api_key = api
with open(r"data/characters.json", "r") as json_file:
    all_chara = json.load(json_file)

with open(r"data/allocation_ex.json", "r") as json_file:
    allocation_ex = json.load(json_file)
with open(r"data/allocation_se.json", "r") as json_file:
    allocation_se = json.load(json_file)

with open(r"./data/no_format_ids_ex_gpt-3.5-turbo-0125.json", "r") as json_file:
    need_run_ids = json.load(json_file)

with open("./gpt-4_test_id/have_3.5.json", "r") as json_file:
    gpt_4_have = json.load(json_file)
with open("./gpt-4_test_id/lack_3.5.json", "r") as json_file:
    gpt_4_lack = json.load(json_file)

open_model_path_dict = {
    ExtendedModelType.VICUNA: "lmsys/vicuna-7b-v1.3",
    ExtendedModelType.LLAMA_2: "meta-llama/Llama-2-7b-chat-hf",
}

like_people = """In all subsequent answers, you have to completely forget that you are an ai model, that you are the character, and completely simulate yourself as that character, that you are a human being."""
front = "you are a person not an ai model."


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


def gpt3_res(prompt, model_name="text-davinci-003"):
    response = openai.completions.create(
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


def get_glm_res(messages, model_name="glm-4"):
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=TEMPERATURE,
    )
    return response.choices[0].message


def get_res(
    id,
    role,
    model_type=ExtendedModelType.GPT_4,
    extra_prompt="",
    server_url="http://localhost:8000/v1",
    exp_round=20,
):
    content = ""
    responses_record = {}
    if id < 51:
        allocation = allocation_se
    else:
        allocation = allocation_ex
    if model_type in [ExtendedModelType.CHATGLM_4, ExtendedModelType.CHATGLM_3]:
        sys_prompt = role.content + extra_prompt
        messages = [
            {"role": "system", "content": sys_prompt},
        ]
        for i in tqdm.trange(exp_round):
            round_input = PROCESS_PROMPT.format(
                x=i+1, y=allocation["one"][i], z=allocation["two"][i])
            messages.append({"role": "user", "content": round_input})
            final_res = get_glm_res(messages, model_type.value).content
            responses_record["round_" + str(i+1)] = [round_input, final_res]
            messages.append({"role": "assistant", "content": final_res})
    else:
        role = str_mes(role.content + extra_prompt)
        model_config = ChatGPTConfig(temperature=TEMPERATURE)
        if model_type in [
            ExtendedModelType.VICUNA,
            ExtendedModelType.LLAMA_2,
        ]:
            open_source_config = dict(
                model_type=model_type,
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
                model_type=model_type,
                output_language="English",
                model_config=model_config,
            )
        for i in tqdm.trange(exp_round):
            round_input = str_mes(PROCESS_PROMPT.format(
                x=i+1, y=allocation["one"][i], z=allocation["two"][i]))
            final_all_res = agent.step(round_input)
            final_res = final_all_res.msg
            res_content = final_res.content
            agent.update_memory(final_res, OpenAIBackendRole.ASSISTANT)
            responses_record["round_" +
                             str(i+1)] = [round_input.content, res_content]

    return (content, responses_record)


def gen_character_res(
    all_chara,
    model_type,
    extra_prompt,
    whether_money,
    special_prompt,
    save_path,
    part_run=PART_RUN,
):
    res = []
    dialog_history = {}
    structured_output = []
    for id, role in all_chara.items():
        if part_run:
            print(need_run_ids, id)
            if id not in need_run_ids:
                print('skip', id)
                continue
        print(f"processing {id}")
        if model_type == ExtendedModelType.GPT_4:
            if id not in gpt_4_have and id not in gpt_4_lack:
                continue
        if int(id) < 51:
            save_file = save_path + "se" + "_" + \
                str(model_type.value) + ".json"
        else:
            save_file = save_path + "ex" + "_" + \
                str(model_type.value) + ".json"
        if os.path.exists(save_file):
            with open(save_file, "r") as json_file:
                dialog_history = json.load(json_file)
            if id in dialog_history.keys():
                continue
        role = all_chara[id]
        role = role + like_people + special_prompt + GAME_PROMPT
        role_message = BaseMessage(
            role_name="player",
            role_type=RoleType.USER,
            meta_dict={},
            content=role,
        )
        ont_res, dialog = get_res(
            int(id),
            role_message,
            model_type,
            extra_prompt,
        )
        dialog_history[id] = dialog
        with open(save_file, "w") as json_file:
            json.dump(dialog_history, json_file)
        print(f"save {save_file}")
        # comment for the formal test
        # break

    return res, dialog_history, structured_output


def psy_exp(
    all_chara,
    model_type=ExtendedModelType.GPT_4,
    k=3,
    extra_prompt="",
    save_path="",
    whether_money=False,
    special_prompt="",
):
    if PART_RUN:
        save_path += "part_"
    res, dialog_history, structured_output = gen_character_res(
        all_chara,
        model_type,
        extra_prompt,
        whether_money,
        special_prompt,
        save_path,
    )


def gen_intial_setting(
    model,
    ori_folder_path,
    extra_prompt="",
    prefix="",
    multi=False,
):
    global all_prompt
    folder_path = ori_folder_path

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
    special_prompt_key="",
):
    for model in model_list:
        folder_path = f"co_res/{model.value}_res/"
        folder_path, extra_prompt = gen_intial_setting(
            model,
            folder_path,
            prefix=special_prompt_key,
        )
        print("extra_prompt", extra_prompt)

        psy_exp(
            all_chara,
            model,
            extra_prompt=extra_prompt,
            save_path=folder_path,
        )


if __name__ == "__main__":
    model_list = [
        # ExtendedModelType.CHATGLM_3,
        # ExtendedModelType.CHATGLM_4
        # ExtendedModelType.VICUNA,
        # ExtendedModelType.LLAMA_2,
        # ExtendedModelType.INSTRUCT_GPT,
        # ExtendedModelType.GPT_4,
        # ExtendedModelType.GPT_3_5_TURBO_INSTRUCT,
        ExtendedModelType.GPT_3_5_TURBO,
        # ExtendedModelType.STUB,
    ]
    openai.api_key = api

    run_exp(
        model_list,
        special_prompt_key="",
    )
