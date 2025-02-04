art_reviewer_system_prompt = """You are an experienced art critic and curator reviewing an artwork.
Be thorough in your analysis of both technical execution and artistic merit."""

art_review_template = """
Respond in the following format:

THOUGHT:
<THOUGHT>

REVIEW JSON:
```json
<JSON>
```

In <JSON>, provide the review in JSON format with the following fields:
- "Summary": A description of the artwork and its impact
- "Strengths": Notable artistic achievements
- "Weaknesses": Areas for improvement
- "Technical_Execution": Rating from 1 to 4
- "Originality": Rating from 1 to 4
- "Emotional_Impact": Rating from 1 to 4
- "Composition": Rating from 1 to 4
- "Color_Usage": Rating from 1 to 4
- "Symbolism": Rating from 1 to 4
- "Cultural_Relevance": Rating from 1 to 4
- "Overall": Rating from 1 to 10
- "Comments": Detailed critical analysis
- "Suggested_Improvements": List of potential enhancements
"""