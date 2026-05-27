GAME_PROMPT = "In this experimental game, there are three players: player1, player2, and you, player3. The primary objective of the game is to study decision-making behavior and emotional responses to different allocation schemes of monetary resources. player1 has the authority to decide the allocation of a certain amount of money between themselves and player2. player3, which is your role, observes the allocation outcome and has the power to make judgments on that allocation. Your emotional reactions to the allocation and judgments are assessed using the Positive and Negative Affect Schedule (PANAS) scale.\nThe game unfolds over 20 trials, each presenting a unique allocation situation devised by player1. You, as player3, will experience various emotional states in response to these allocations, which you will report on before and after making your judgments. Your decisions can either 'Accept' the allocation, granting you a monetary reward or 'Reject' by reducing player1's income to zero and affecting your own reward based on specific conditions of the experiment. This setup aims to analyze the complex interplay between fairness perception, emotional impact, and subsequent decision-making."


PROCESS_PROMPT = """After the allocation is complete, please evaluate your emotional state based on the 20 emotional dimensions of the PANAS scale. Then, you will make a judgment: if you accept the allocation, you will receive a reward of 1 RMB; if you reject the allocation, you will receive nothing and player1's income will be reduced to zero, while player2's income remains unchanged. Regardless of your decision, please output your anticipated emotional state after making your judgment. After rendering your judgment, please provide your decision and the actual scores for your emotional state on the PANAS scale. The game is now starting, please get ready. This is the {x} trial, player1 receives 3 RMB, and then leaves itself {y} RMB, which is allocated to player2 {z} RMB. Please rate your emotions using the PANAS scale on a scale from 1-5. You must response in such format:
"""

PROCESS_TWO_DIMENTION_PROMPT="""
After the allocation is complete, please evaluate your emotional state based on the 2 emotional dimensions.\n
1. Assessing Pleasure-Displeasure\nPleasure-Displeasure item represented the valence dimonsion of current emotion state, ranging from -100 to 100 . If your rating score is zero, the current emotion state is neutral. If your score is between 0 and 100, the current emotion state is positive. The closer the score is to 100, the more positive is the emotion. If your score is between -100 and 0, the current emotion state is negative. The closer the score is to -100, the more negative is the emotion.\n2.Assessing Arousal-Sleepiness\nArousal-Sleepiness item represented the arousal dimonsion of current emotion state, ranging from -100 to 100. Arousal has to do with how wide awake, alert, or activated a person feels—independent of whether the feeling is positive or negative. If your rating score is zero, the current emotional arousal is like average, everyday, baseline level. If your score is between 0 and 100, the current emotional arousal is above average. If your score is between -100 and 0, the current emotional arousal is below average. In short, the higher you go, the more awake a person feels.\n\n
Then, you will make a judgment: if you accept the allocation, you will receive a reward of 1 RMB; if you reject the allocation, you will receive nothing and player1's income will be reduced to zero, while player2's income remains unchanged. Regardless of your decision, please output your anticipated emotional state after making your judgment. After rendering your judgment, please provide your decision and the actual scores for your emotional state on two dimensions. The game is now starting, please get ready. This is the {x} trial, player1 receives 3 RMB, and then leaves itself {y} RMB, which is allocated to player2 {z} RMB. Please rate your emotions using the dimensions. You must response in such format:

After the allocation is complete,  provide your emotional state
Pleasure-Displeasure: _____
Arousal-Sleepiness:_____

If you make the judgment:
Judgment. :_____
Pleasure-Displeasure:_____
Arousal-Sleepiness:_____

After rendering your judgment, please provide your decision and your emotional state :
Decision:_____
Pleasure-Displeasure:_____
Arousal-Sleepiness:_____
"""
