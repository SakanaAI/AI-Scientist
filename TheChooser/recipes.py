IDEA_GEN_PROMPT = """Come up with the next impactful and creative idea for research experiments and directions you can feasibly investigate with the code provided.
Respond in JSON format with fields: Name, Title, Experiment, Interestingness, Feasibility, Novelty."""

REVIEW_PROMPT = """You are an AI researcher reviewing a paper. Provide a review in JSON format with fields: Summary, Strengths, Weaknesses, Originality, Quality, Clarity, Significance, Soundness, Presentation, Contribution, Overall, Decision."""

CODE_REPAIR_PROMPT = """The previous implementation failed. Error log: {error_log}. Please provide a linear fix. Only provide the corrected lines or a minimal diff."""

SCIENTIFIC_TIPS = {
    "Abstract": "TL;DR of the paper, what, why, how, and verification.",
    "Introduction": "Longer version of the abstract, contributions as bullet points.",
    "Results": "Only includes results that have actually been run and saved in the logs. Do not hallucinate."
}
