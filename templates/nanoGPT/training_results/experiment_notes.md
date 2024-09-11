# nanoGPT Experiment Notes

## Experiment Setup
- Model: nanoGPT
- Dataset: Shakespeare
- Training iterations: 2020 (interrupted)
- Hardware: CPU (MacBook Air)

## Modifications to Original Script
- No significant modifications were made to the original experiment.py script.

## Challenges Faced

1. CPU Training:
   - Training was performed on CPU, which resulted in slower iteration times compared to GPU training.
   - Iteration times varied significantly, ranging from ~5000ms to over 10000ms per iteration.

2. Training Interruption:
   - The training process was interrupted at iteration 2020, possibly due to a KeyboardInterrupt.
   - This prevented the completion of the full training cycle and the saving of final results.

3. Result Saving:
   - The interruption led to no checkpoint files or final results being saved.
   - This made it impossible to resume training or evaluate the final model state.

4. Plotting Issues:
   - Attempted to run plot.py, but it failed due to missing 'final_info.json' file.

## Lessons Learned

1. Importance of Checkpointing:
   - Regular checkpointing is crucial for long-running experiments to allow resumption and prevent data loss.

2. Robust Error Handling:
   - Implement better error handling to ensure partial results are saved even in case of interruptions.

3. Progress Logging:
   - More detailed progress logging could provide better insights into training dynamics.

4. Hardware Considerations:
   - CPU training is feasible but time-consuming. Consider using GPU resources for faster iterations in future experiments.

