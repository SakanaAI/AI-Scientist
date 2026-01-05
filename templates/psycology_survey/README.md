# psychology Survey template

The psychology survey template aims to explore using the AI-scientist
for simulating psychology studies using LLM personas, and then surveying them for analysis.

[LLMs has been used](https://arxiv.org/pdf/2304.03442) to simulate human behaviours. There is potentially value in recreating historical psychology studies using LLM personas given that LLMs encode a large array of human behaviours.

This template does so by templaziting survey based approaches in psychology studies.

To use this template, we need to define 3 components.

1) survey.json. This defines who are we surveying, what questions are we asking, and what analysis is to be done.
2) Personas. These are yaml files that describe virtual personas to be surveyed.
3) analysis.py. This file contains logic that maps analysis types in survey.json to python functions.  

As an example, ideas.json is populated with an idea that tries to reproduce the landmark psychology paper [Happiness and unhappiness in the East and West](https://www.researchgate.net/profile/Yukiko-Uchida/publication/26716010_Happiness_and_Unhappiness_in_East_and_West_Themes_and_Variations/links/0c960525e02a7d5940000000/Happiness-and-Unhappiness-in-East-and-West-Themes-and-Variations.pdf). This paper examined the differences in themes of happy and unhappy emotions in Japan and the United states. The template aims to reproduce the paper using LLM personas set in Japan and US.


## Happiness and unhappiness in the east and west example

Run the following to reproduce this experiment. Example paper is included. Note that this experiment doesn't need GPUs to run.

```
python launch_scientist.py --model "gemini-2.5-pro" --experiment psychology_survey --skip-idea-generation --skip-novelty-check --per-experiment-files analyze.py,personas.py,survey.py,data/personas/jp.yaml,data/personas/us.yaml,data/survey.json
```
