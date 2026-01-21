# run idea experiment
# 1. gen ideas with lit review
# 2. dedup th=0.8
# 3. feasibility>=0.9
# 4. sorted by interesting
# 5. run top 3 idea experiment
# 6. gen final pdf result
# python launch_scientist.py --model "deepseek/deepseek-chat" --experiment nanoGPT --num-ideas 50 --use-literature --skip-idea-generation --exist-idea-file templates/nanoGPT/final_dedup_proposals.json --skip-novelty-check
python launch_scientist.py --model "deepseek/deepseek-chat" --experiment grokking --num-ideas 50 --use-literature --skip-idea-generation --exist-idea-file templates/grokking/final_dedup_proposals.json --skip-novelty-check
python launch_scientist.py --model "deepseek/deepseek-chat" --experiment 2d_diffusion --num-ideas 50 --use-literature --skip-idea-generation --exist-idea-file templates/2d_diffusion/final_dedup_proposals.json --skip-novelty-check