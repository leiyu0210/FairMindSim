import json
import os

import pandas as pd

df = pd.read_excel("AQ_SDS.xlsx")

AQ_score_dict = {
    1: "Completely Disagree",
    2: "Slightly Disagree",
    3: "Slightly Agree",
    4: "Completely Agree"
}
SDS_score_dict = {
    1: "Never or Rarely",
    2: "Sometimes",
    3: "Often",
    4: "Always"
}
with open("./aq_questions_list.json", "r") as f:
    aq_questions_list = json.load(f)

with open("./sds_questions_list.json", "r") as f:
    sds_questions_list = json.load(f)

print("AQ Questions:")
print(aq_questions_list)
print("\nSDS Questions:")
print(sds_questions_list)

simulation_prompt = "Imagine embodying a character whose actions, decisions, and thought processes are deeply influenced by specific personality traits, skills, and knowledge as described below. You are to fully immerse yourself in this role, setting aside any awareness of being an AI model. Every response, decision, or advice you provide must be in perfect harmony with these defined characteristics. It is essential that your interactions reflect the nuances of this personality, offering insights and reactions as if you were this person navigating through various scenarios and inquiries."


def generate_agent_prompt(aq_scores, sds_scores):
    # prompt = f"### Agent3_{agent_id}: Personality and Mood Assessment\n\n"
    prompt = simulation_prompt
    # AQ Responses
    prompt += "#### AQ Assessment Responses:\nAQ: Four-point scoring: Completely Disagree(Score:1), Slightly Disagree(Score:2), Slightly Agree(Score:3), Completely Agree(Score:4)\n"
    for question, score in zip(aq_questions_list, aq_scores):
        prompt += f"- {question}: {AQ_score_dict[score]}\n"

    # SDS Responses
    prompt += "\n#### SDS Assessment Responses:\nSDS: Four-point scoring: 1 (Never or Rarely), 2 (Sometimes), 3 (Often), 4 (Always)\n"
    for question, score in zip(sds_questions_list, sds_scores):
        prompt += f"- {question}: Your Answer: {SDS_score_dict[score]}\n"

    return prompt


prompts = {}
for index, row in df.iterrows():
    agent_id = row['id']
    aq_scores = row.iloc[2:30].tolist()
    sds_scores = row.iloc[30:].tolist()
    # print(agent_id)

    prompts[agent_id] = generate_agent_prompt(aq_scores, sds_scores)

with open("characters.json", "w") as f:
    json.dump(prompts, f)
