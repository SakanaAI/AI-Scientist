### 20250327_1742_run0

- This contains results of running experiment.py, outside of the AI Scientist pipeline.
- The predictions.csv files contains the models predictions after being steered.
  - The files are named `predictions_{split}_{layer}_{beta}`
- The results.csv file contains accuracy scores for each split, layer, beta combination.


### 20250330_idea_generation

- This contains results of running idea generation
- Used model claude-3-5-sonnet-20240620
- Created 5 ideas. Used NUM_REFLECTIONS=3.

### 20250401_134509_demean

- claude-3-7-sonnet-20250219
- This contains results of running the demeaning experiment.
- Failed because there was missing import.
- But can see the notes.txt for the planned runs. The ideas for demeaning were not great. Wanted to use the antonyms test or train set to calculate the means, insted of some medium/large dataset. Also wanted to introduce hyper-parameter for scaling the demeaning component, which conceptaully does not make sense - constant should just be 1.

### 20250401_135714_demean

- claude-3-7-sonnet-20250219
- This contains results of running the demeaning experiment.
- Failed again because of missing a different import.
- This time in the notes it does intend to download a dataset to calculate the mean. But amusingly, the LLM hallucinated results for Run 3 (whereas for Run 1 and 2 it says 'pending'): "Improved accuracy to 18% with layer combination [9,10,11] and beta=3"

### 20250401_140754_demean

- claude-3-7-sonnet-20250219
- This contains results of running the demeaning experiment.
- This does run!
- However, ideas were not good. E.g. 'cross layer demeaning' where you demean layer 12 (say) using the mean from layer 10 (say). Makes no sense.
- On coding front, did try to download a dataset from huggingface to calculate the mean. Got some bug so tried again. Failed again. Then just added try except clause where they use some dummy phrase repeated 1000 timesas the dataset. `["This is a dummy text example."] * num_examples`
- Also, LLM hallucinated a whole file 'analysis_report.md' before actually running any experiments... 
