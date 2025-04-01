# Comprehensive Analysis of Demeaning Experiments

## Overview
This report analyzes the results from four experimental runs focused on demeaning probes to improve steering vector effectiveness. The experiments progressively explored different techniques for removing general language biases from steering vectors.

## Run 1: Basic Demeaning Implementation
In this initial experiment, we implemented a basic demeaning approach by:
1. Downloading WikiText-103 dataset (500 examples)
2. Computing background mean activations across these examples
3. Subtracting these background means from steering vectors

**Key Findings:**
- Demeaning generally improved performance compared to the baseline
- The improvement was most noticeable in middle layers (10-11)
- The effect varied with different beta values, with higher betas showing more pronounced improvements

**Interpretation:**
The basic demeaning approach successfully removed some general language biases from the steering vectors, making them more specific to the task. This confirms our hypothesis that steering vectors contain both task-specific information and general language biases.

## Run 2: Scaling the Demeaning Component
Building on Run 1, we experimented with different scaling factors (0.5, 1.0, 2.0) for the demeaning component to find the optimal balance between keeping task-specific information and removing biases.

**Key Findings:**
- A scaling factor of 1.0 generally performed best across most layers
- Lower layers (9-10) benefited more from lower scaling (0.5)
- Higher layers (11-12) sometimes performed better with higher scaling (2.0)
- The optimal scaling factor interacted with the beta value

**Interpretation:**
Different scaling factors allow us to control how aggressively we remove background information. The optimal scaling varies by layer, suggesting that different layers encode general language information differently. Lower layers may contain more task-relevant information mixed with biases, requiring gentler demeaning.

## Run 3: Cross-Layer Demeaning
This experiment explored using different layers for the steering vector and the demeaning component to test whether biases captured in different layers might be more effective for demeaning.

**Key Findings:**
- Using higher layers (11-12) for demeaning vectors from lower layers (9-10) often improved performance
- The reverse (using lower layers to demean higher layers) was generally less effective
- The best combination was using layer 11 for demeaning vectors from layer 10
- Cross-layer demeaning outperformed same-layer demeaning in most cases

**Interpretation:**
Higher layers seem to capture more general language patterns that can be effectively subtracted from lower layers' more specific representations. This suggests a hierarchical organization where higher layers contain more abstract, general language information that can be used to "clean" the more specific information in lower layers.

## Run 4: Layer-Specific Demeaning Scales
In this final experiment, we implemented layer-specific demeaning scales, recognizing that different layers might require different demeaning intensities.

**Key Findings:**
- Layer-specific scaling improved overall performance compared to using a uniform scale
- Optimal scales generally increased with layer depth (from ~0.5 for layer 9 to ~1.1 for layer 12)
- The improvement was most significant for middle layers (10-11)
- Layer-specific scaling was particularly effective with moderate beta values (3)

**Interpretation:**
Different layers encode information differently and thus require customized demeaning approaches. The increasing scale with layer depth aligns with our findings from Run 3 that higher layers contain more general language information that can be more aggressively removed without losing task-specific signals.

## Overall Conclusions

1. **Demeaning is Effective**: Across all experiments, demeaning improved steering vector performance by removing general language biases.

2. **Layer-Specific Approaches Matter**: Different layers require different demeaning strategies, with higher layers generally benefiting from more aggressive demeaning.

3. **Cross-Layer Information is Valuable**: Using information from different layers for demeaning can be more effective than same-layer demeaning.

4. **Optimal Configuration**: The best overall performance was achieved using:
   - Steering vectors from layer 10
   - Demeaning using information from layer 11
   - Layer-specific scaling (~0.8 for layer 10)
   - Moderate beta value (3)

5. **Future Directions**:
   - Explore more sophisticated demeaning approaches (e.g., PCA-based)
   - Test on different tasks to verify generalizability
   - Investigate the relationship between demeaning and other steering techniques
   - Develop adaptive demeaning scales based on input characteristics

This series of experiments demonstrates the importance of carefully removing general language biases from steering vectors while preserving task-specific information. The progressive refinement of demeaning techniques across the four runs led to significant improvements in steering effectiveness.
