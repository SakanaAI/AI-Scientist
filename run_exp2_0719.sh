# run idea experiment
# 1. gen ideas with lit review
# 2. dedup th=0.8
# 3. feasibility>=0.9
# 4. sorted by interesting
# 5. run top 3 idea experiment
# 6. gen final pdf result
# python launch_scientist_exp2.py --model "deepseek/deepseek-chat" --experiment nanoGPT --num-ideas 10 --use-literature --run-idea-dedup
python launch_scientist_exp2.py --model "deepseek/deepseek-chat" --experiment grokking --num-ideas 10 --use-literature --run-idea-dedup
python launch_scientist_exp2.py --model "deepseek/deepseek-chat" --experiment 2d_diffusion --num-ideas 10 --use-literature --run-idea-dedup