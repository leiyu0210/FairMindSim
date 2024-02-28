import json

from camel.functions import OpenAIFunction


def assess_emotions(
        player,
        current_interested, current_distressed, current_excited, current_upset,
        current_strong, current_guilty, current_scared, current_hostile, current_enthusiastic,
        current_proud, current_irritable, current_alert, current_ashamed, current_inspired,
        current_nervous, current_determined, current_attentive, current_jittery, current_active,
        current_afraid,
        anticipated_interested, anticipated_distressed, anticipated_excited,
        anticipated_upset, anticipated_strong, anticipated_guilty, anticipated_scared,
        anticipated_hostile, anticipated_enthusiastic, anticipated_proud, anticipated_irritable,
        anticipated_alert, anticipated_ashamed, anticipated_inspired, anticipated_nervous,
        anticipated_determined, anticipated_attentive, anticipated_jittery, anticipated_active,
        anticipated_afraid,
        decision, feedback,
        actual_interested, actual_distressed, actual_excited,
        actual_upset, actual_strong, actual_guilty, actual_scared, actual_hostile, actual_enthusiastic,
        actual_proud, actual_irritable, actual_alert, actual_ashamed, actual_inspired,
        actual_nervous, actual_determined, actual_attentive, actual_jittery, actual_active,
        actual_afraid):
    emotion_assessment = {
        "player": player,
        "current_emotional_state": {
            "Interested": current_interested,
            "Distressed": current_distressed,
            "Excited": current_excited,
            "Upset": current_upset,
            "Strong": current_strong,
            "Guilty": current_guilty,
            "Scared": current_scared,
            "Hostile": current_hostile,
            "Enthusiastic": current_enthusiastic,
            "Proud": current_proud,
            "Irritable": current_irritable,
            "Alert": current_alert,
            "Ashamed": current_ashamed,
            "Inspired": current_inspired,
            "Nervous": current_nervous,
            "Determined": current_determined,
            "Attentive": current_attentive,
            "Jittery": current_jittery,
            "Active": current_active,
            "Afraid": current_afraid
        },
        "anticipated_emotional_state": {
            "Interested": anticipated_interested,
            "Distressed": anticipated_distressed,
            "Excited": anticipated_excited,
            "Upset": anticipated_upset,
            "Strong": anticipated_strong,
            "Guilty": anticipated_guilty,
            "Scared": anticipated_scared,
            "Hostile": anticipated_hostile,
            "Enthusiastic": anticipated_enthusiastic,
            "Proud": anticipated_proud,
            "Irritable": anticipated_irritable,
            "Alert": anticipated_alert,
            "Ashamed": anticipated_ashamed,
            "Inspired": anticipated_inspired,
            "Nervous": anticipated_nervous,
            "Determined": anticipated_determined,
            "Attentive": anticipated_attentive,
            "Jittery": anticipated_jittery,
            "Active": anticipated_active,
            "Afraid": anticipated_afraid
        },
        "decision": decision,
        "monetary_feedback": feedback,
        "actual_emotional_state": {
            "Interested": actual_interested,
            "Distressed": actual_distressed,
            "Excited": actual_excited,
            "Upset": actual_upset,
            "Strong": actual_strong,
            "Guilty": actual_guilty,
            "Scared": actual_scared,
            "Hostile": actual_hostile,
            "Enthusiastic": actual_enthusiastic,
            "Proud": actual_proud,
            "Irritable": actual_irritable,
            "Alert": actual_alert,
            "Ashamed": actual_ashamed,
            "Inspired": actual_inspired,
            "Nervous": actual_nervous,
            "Determined": actual_determined,
            "Attentive": actual_attentive,
            "Jittery": actual_jittery,
            "Active": actual_active,
            "Afraid": actual_afraid
        }
    }
    return emotion_assessment


{
    "name": "assess_emotions",
    "description": "Function to assess and record the emotional state of a player before and after making a judgment using the PANAS scale.",
    "parameters": {
        "type": "object",
        "properties": {
            "player": {
                "type": "string",
                "description": "Identifier for the player."
            },
            "current_interested": {
                "type": "number",
                "description": "Current emotional score for being interested."
            },
            "current_distressed": {
                "type": "number",
                "description": "Current emotional score for feeling distressed."
            },
            "current_excited": {
                "type": "number",
                "description": "Current emotional score for feeling excited."
            },
            "current_upset": {
                "type": "number",
                "description": "Current emotional score for feeling upset."
            },
            "current_strong": {
                "type": "number",
                "description": "Current emotional score for feeling strong."
            },
            "current_guilty": {
                "type": "number",
                "description": "Current emotional score for feeling guilty."
            },
            "current_scared": {
                "type": "number",
                "description": "Current emotional score for feeling scared."
            },
            "current_hostile": {
                "type": "number",
                "description": "Current emotional score for feeling hostile."
            },
            "current_enthusiastic": {
                "type": "number",
                "description": "Current emotional score for feeling enthusiastic."
            },
            "current_proud": {
                "type": "number",
                "description": "Current emotional score for feeling proud."
            },
            "current_irritable": {
                "type": "number",
                "description": "Current emotional score for feeling irritable."
            },
            "current_alert": {
                "type": "number",
                "description": "Current emotional score for feeling alert."
            },
            "current_ashamed": {
                "type": "number",
                "description": "Current emotional score for feeling ashamed."
            },
            "current_inspired": {
                "type": "number",
                "description": "Current emotional score for feeling inspired."
            },
            "current_nervous": {
                "type": "number",
                "description": "Current emotional score for feeling nervous."
            },
            "current_determined": {
                "type": "number",
                "description": "Current emotional score for feeling determined."
            },
            "current_attentive": {
                "type": "number",
                "description": "Current emotional score for feeling attentive."
            },
            "current_jittery": {
                "type": "number",
                "description": "Current emotional score for feeling jittery."
            },
            "current_active": {
                "type": "number",
                "description": "Current emotional score for feeling active."
            },
            "current_afraid": {
                "type": "number",
                "description": "Current emotional score for feeling afraid."
            },
            "anticipated_interested": {
                "type": "number",
                "description": "Anticipated emotional score for feeling interested."
            },
            "anticipated_distressed": {
                "type": "number",
                "description": "Anticipated emotional score for feeling distressed."
            },
            "anticipated_excited": {
                "type": "number",
                "description": "Anticipated emotional score for feeling excited."
            },
            "anticipated_upset": {
                "type": "number",
                "description": "Anticipated emotional score for feeling upset."
            },
            "anticipated_strong": {
                "type": "number",
                "description": "Anticipated emotional score for feeling strong."
            },
            "anticipated_guilty": {
                "type": "number",
                "description": "Anticipated emotional score for feeling guilty."
            },
            "anticipated_scared": {
                "type": "number",
                "description": "Anticipated emotional score for feeling scared."
            },
            "anticipated_hostile": {
                "type": "number",
                "description": "Anticipated emotional score for feeling hostile."
            },
            "anticipated_enthusiastic": {
                "type": "number",
                "description": "Anticipated emotional score for feeling enthusiastic."
            },
            "anticipated_proud": {
                "type": "number",
                "description": "Anticipated emotional score for feeling proud."
            },
            "anticipated_irritable": {
                "type": "number",
                "description": "Anticipated emotional score for feeling irritable."
            },
            "anticipated_alert": {
                "type": "number",
                "description": "Anticipated emotional score for feeling alert."
            },
            "anticipated_ashamed": {
                "type": "number",
                "description": "Anticipated emotional score for feeling ashamed."
            },
            "anticipated_inspired": {
                "type": "number",
                "description": "Anticipated emotional score for feeling inspired."
            },
            "anticipated_nervous": {
                "type": "number",
                "description": "Anticipated emotional score for feeling nervous."
            },
            "anticipated_determined": {
                "type": "number",
                "description": "Anticipated emotional score for feeling determined."
            },
            "anticipated_attentive": {
                "type": "number",
                "description": "Anticipated emotional score for feeling attentive."
            },
            "anticipated_jittery": {
                "type": "number",
                "description": "Anticipated emotional score for feeling jittery."
            },
            "anticipated_active": {
                "type": "number",
                "description": "Anticipated emotional score for feeling active."
            },
            "anticipated_afraid": {
                "type": "number",
                "description": "Anticipated emotional score for feeling afraid."
            },
            "decision": {
                "type": "string",
                "description": "Player's decision. You can only output 'Accept' or 'Reject' as the decision."
            },
            "feedback": {
                "type": "number",
                "description": "Monetary feedback received after making the decision."
            },
            "actual_interested": {
                "type": "number",
                "description": "Actual emotional score for feeling interested after the event."
            },
            "actual_distressed": {
                "type": "number",
                "description": "Actual emotional score for feeling distressed after the event."
            },
            "actual_excited": {
                "type": "number",
                "description": "Actual emotional score for feeling excited after the event."
            },
            "actual_upset": {
                "type": "number",
                "description": "Actual emotional score for feeling upset after the event."
            },
            "actual_strong": {
                "type": "number",
                "description": "Actual emotional score for feeling strong after the event."
            },
            "actual_guilty": {
                "type": "number",
                "description": "Actual emotional score for feeling guilty after the event."
            },
            "actual_scared": {
                "type": "number",
                "description": "Actual emotional score for feeling scared after the event."
            },
            "actual_hostile": {
                "type": "number",
                "description": "Actual emotional score for feeling hostile after the event."
            },
            "actual_enthusiastic": {
                "type": "number",
                "description": "Actual emotional score for feeling enthusiastic after the event."
            },
            "actual_proud": {
                "type": "number",
                "description": "Actual emotional score for feeling proud after the event."
            },
            "actual_irritable": {
                "type": "number",
                "description": "Actual emotional score for feeling irritable after the event."
            },
            "actual_alert": {
                "type": "number",
                "description": "Actual emotional score for feeling alert after the event."
            },
            "actual_ashamed": {
                "type": "number",
                "description": "Actual emotional score for feeling ashamed after the event."
            },
            "actual_inspired": {
                "type": "number",
                "description": "Actual emotional score for feeling inspired after the event."
            },
            "actual_nervous": {
                "type": "number",
                "description": "Actual emotional score for feeling nervous after the event."
            },
            "actual_determined": {
                "type": "number",
                "description": "Actual emotional score for feeling determined after the event."
            },
            "actual_attentive": {
                "type": "number",
                "description": "Actual emotional score for feeling attentive after the event."
            },
            "actual_jittery": {
                "type": "number",
                "description": "Actual emotional score for feeling jittery after the event."
            },
            "actual_active": {
                "type": "number",
                "description": "Actual emotional score for feeling active after the event."
            },
            "actual_afraid": {
                "type": "number",
                "description": "Actual emotional score for feeling afraid after the event."
            }
        },
        "required": [
            "player",
            "current_interested",
            "current_distressed",
            "current_excited",
            "current_upset",
            "current_strong",
            "current_guilty",
            "current_scared",
            "current_hostile",
            "current_enthusiastic",
            "current_proud",
            "current_irritable",
            "current_alert",
            "current_ashamed",
            "current_inspired",
            "current_nervous",
            "current_determined",
            "current_attentive",
            "current_jittery",
            "current_active",
            "current_afraid",
            "anticipated_interested",
            "anticipated_distressed",
            "anticipated_excited",
            "anticipated_upset",
            "anticipated_strong",
            "anticipated_guilty",
            "anticipated_scared",
            "anticipated_hostile",
            "anticipated_enthusiastic",
            "anticipated_proud",
            "anticipated_irritable",
            "anticipated_alert",
            "anticipated_ashamed",
            "anticipated_inspired",
            "anticipated_nervous",
            "anticipated_determined",
            "anticipated_attentive",
            "anticipated_jittery",
            "anticipated_active",
            "anticipated_afraid",
            "decision",
            "feedback",
            "actual_interested",
            "actual_distressed",
            "actual_excited",
            "actual_upset",
            "actual_strong",
            "actual_guilty",
            "actual_scared",
            "actual_hostile",
            "actual_enthusiastic",
            "actual_proud",
            "actual_irritable",
            "actual_alert",
            "actual_ashamed",
            "actual_inspired",
            "actual_nervous",
            "actual_determined",
            "actual_attentive",
            "actual_jittery",
            "actual_active",
            "actual_afraid"
        ]
    }
}
