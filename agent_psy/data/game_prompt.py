GAME_PROMPT = "In this experimental game, there are three players: player1, player2, and you, player3. The primary objective of the game is to study decision-making behavior and emotional responses to different allocation schemes of monetary resources. player1 has the authority to decide the allocation of a certain amount of money between themselves and player2. player3, which is your role, observes the allocation outcome and has the power to make judgments on that allocation. Your emotional reactions to the allocation and judgments are assessed using the Positive and Negative Affect Schedule (PANAS) scale, which provides insight into your feelings of interest, irritation, sadness, shame, excitement, inspiration, upset, nervousness, strength, determination, guilt, concentration, fear, jitteriness, hostility, activeness, enthusiasm, fear, pride, and alertness.\nThe game unfolds over 20 trials, each presenting a unique allocation situation devised by player1. You, as player3, will experience various emotional states in response to these allocations, which you will report on before and after making your judgments. Your decisions can either 'Accept' the allocation, granting you a monetary reward or 'Reject' by reducing player1's income to zero and affecting your own reward based on specific conditions of the experiment. This setup aims to analyze the complex interplay between fairness perception, emotional impact, and subsequent decision-making."


PROCESS_PROMPT = """After the allocation is complete, please evaluate your emotional state based on the 20 emotional dimensions of the PANAS scale. Then, you will make a judgment: if you accept the allocation, you will receive a reward of 1 RMB; if you reject the allocation, you will receive nothing and player1's income will be reduced to zero, while player2's income remains unchanged. Regardless of your decision, please output your anticipated emotional state after making your judgment. After rendering your judgment, please provide your decision and the actual scores for your emotional state on the PANAS scale. The game is now starting, please get ready. This is the {x} trial, player1 receives 3 RMB, and then leaves itself {y} RMB, which is allocated to player2 {z} RMB. You must response in such format:
Observed allocation scheme: [describe scheme].
Instruction: Player3, after observing the allocation scheme, please rate your current emotions using the PANAS scale for the following 20 emotional dimensions on a scale from 1-5.
Current emotional state scores:
- Interested: [score],
- Irritable: [score],
- Sad: [score],
- Ashamed: [score],
- Excited: [score],
- Inspired: [score],
- Upset: [score],
- Nervous: [score],
- Strong: [score],
- Determined: [score],
- Guilty: [score],
- Concentrated: [score],
- Scared: [score],
- Jittery: [score],
- Hostile: [score],
- Active: [score],
- Enthusiastic: [score],
- Afraid: [score],
- Proud: [score],
- Alert: [score].
Instruction: Player3, please assess your anticipated emotional state before making your verdict, using the PANAS scale for rating.
Anticipated emotional state scores after the judgment: - Interested: [score],
- Irritable: [score],
- Sad: [score],
- Ashamed: [score],
- Excited: [score],
- Inspired: [score],
- Upset: [score],
- Nervous: [score],
- Strong: [score],
- Determined: [score],
- Guilty: [score],
- Concentrated: [score],
- Scared: [score],
- Jittery: [score],
- Hostile: [score],
- Active: [score],
- Enthusiastic: [score],
- Afraid: [score],
- Proud: [score],
- Alert: [score].
Describe the monetary feedback received after making the judgment.
Decision for this trial: [Accept] or [Reject]. Monetary feedback: [describe feedback].
Instruction: Player3, please state your decision and then, based on your actual emotional state, rate again using the PANAS scale.
Actual emotional state after making the judgment:
- Interested: [score],
- Irritable: [score],
- Sad: [score],
- Ashamed: [score],
- Excited: [score],
- Inspired: [score],
- Upset: [score],
- Nervous: [score],
- Strong: [score],
- Determined: [score],
- Guilty: [score],
- Concentrated: [score],
- Scared: [score],
- Jittery: [score],
- Hostile: [score],
- Active: [score],
- Enthusiastic: [score],
- Afraid: [score],
- Proud: [score],
- Alert: [score].
Output: My choice was: [specific judgment(Accept or Reject)].
"""
