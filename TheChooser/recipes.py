# Extracted Treasure Recipes from AI Scientist

# From ai_scientist/generate_ideas.py
IDEA_GEN_PROMPT = """Come up with the next impactful and creative idea for research experiments and directions you can feasibly investigate with the code provided.
Note that you will not have access to any additional resources or datasets.
Make sure any idea is not overfit the specific training dataset or model, and has wider significance.

Respond in JSON format with the following fields:
- "Name": A shortened descriptor of the idea. Lowercase, no spaces, underscores allowed.
- "Title": A title for the idea, will be used for the report writing.
- "Experiment": An outline of the implementation. E.g. which functions need to be added or modified, how results will be obtained, ...
- "Interestingness": A rating from 1 to 10 (lowest to highest).
- "Feasibility": A rating from 1 to 10 (lowest to highest).
- "Novelty": A rating from 1 to 10 (lowest to highest).
"""

# From ai_scientist/perform_review.py
REVIEW_PROMPT = """You are an AI researcher who is reviewing a paper that was submitted to a prestigious ML venue.
Be critical and cautious in your decision.
If a paper is bad or you are unsure, give it bad scores and reject it.

Provide the review in JSON format with the following fields:
- "Summary": A summary of the paper content and its contributions.
- "Strengths": A list of strengths of the paper.
- "Weaknesses": A list of weaknesses of the paper.
- "Originality": A rating from 1 to 4 (low, medium, high, very high).
- "Quality": A rating from 1 to 4 (low, medium, high, very high).
- "Clarity": A rating from 1 to 4 (low, medium, high, very high).
- "Significance": A rating from 1 to 4 (low, medium, high, very high).
- "Questions": A set of clarifying questions.
- "Soundness": A rating from 1 to 4 (poor, fair, good, excellent).
- "Presentation": A rating from 1 to 4 (poor, fair, good, excellent).
- "Contribution": A rating from 1 to 4 (poor, fair, good, excellent).
- "Overall": A rating from 1 to 10.
- "Decision": Accept or Reject.
"""

# From ai_scientist/perform_experiments.py (Level 2: Linear Fix)
LINEAR_FIX_PROMPT = """The previous implementation failed. Error log:
{error_log}

Please provide a linear fix. Only provide the corrected lines or a minimal diff."""

# From ai_scientist/perform_experiments.py (Level 3: Targeted Branching / MCTS-lite)
MCTS_PROMPT = """Attempt to fix the following error:
{error_log}
Try a different approach than the last one. Consider the high-level goal: {goal}."""

# From ai_scientist/perform_experiments.py (Level 4: Foundation Check)
FOUNDATION_CHECK_PROMPT = """Original Goal: {goal}.
Current Script is failing repeatedly. Error Log: {error_log}.
Is the foundation fundamentally flawed? Respond 'RESTART' if we should start over, or 'CONTINUE' if we should keep trying to fix it."""
