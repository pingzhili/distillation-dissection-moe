# MoE Distillation Analysis: Findings Report

## Overview

This report summarizes the findings of our in-depth analysis of routing patterns in Mixture of Experts (MoE) models, specifically investigating whether routing patterns can be used to detect if a model has been distilled from another model. We compared the following models:

1. **olmoe_base**: The base OLMoE-1B-7B-0125 model (pretrained)
2. **olmoe_distill**: An OLMoE model fine-tuned by distillation from Moonlight
3. **olmoe_original**: An OLMoE model fine-tuned on the original dataset (no distillation)
4. **moonlight**: The Moonlight-16B-A3B-Instruct model (which was used as the teacher model)

## Key Findings

### 1. Routing Pattern Similarities

Our analysis of routing patterns across models revealed several important insights:

- **Distillation Signature**: The distilled model (olmoe_distill) shows distinct routing patterns that are influenced by the teacher model (moonlight), suggesting that distillation transfers not only the output probabilities but also affects how the MoE routes tokens internally.

- **Expert Selection Changes**: Comparing the original OLMoE and the distilled version, we found significant changes in expert selection in middle and later layers, while early layers remained relatively stable. This suggests that distillation primarily affects higher-level representations.

- **Top-1 Expert Selection**: The most significant differences appear in the primary expert selection (top-1 expert), with moderate changes in secondary experts. This indicates that distillation influences the primary routing decisions more strongly than the backup experts.

### 2. Dimension-Level Analysis

Analyzing the changes at the dimension level revealed:

- **Specific Dimensions Change More**: Certain dimensions consistently show larger changes across multiple layers after distillation, suggesting these dimensions may be more important for transferring the teacher model's capabilities.

- **Pattern of Dimensional Changes**: The pattern of which dimensions change significantly differs between normal fine-tuning and distillation, potentially providing a fingerprint for detecting distilled models.

- **Dimensional Clusters**: We identified clusters of dimensions that change together during distillation, suggesting coordinated changes in semantic representation.

### 3. Token-Level Analysis

Examining the token-level embeddings provided fine-grained insights:

- **Token Routing Correlation**: We observed a correlation between changes in token embeddings and changes in routing decisions. Tokens with more significant embedding changes also tend to have more different expert selection patterns.

- **Sequence-Specific Effects**: Distillation affects different types of sequences differently. Some sequences maintained similar routing patterns, while others showed dramatic shifts, suggesting content-dependent distillation effects.

- **Context Dependency**: The routing changes for a given token depend not only on the token itself but also on its context, indicating that distillation affects the contextual understanding of tokens.

### 4. Differences Between Distillation and Regular Fine-tuning

A key objective was to distinguish between changes due to distillation versus regular fine-tuning:

- **Routing Distribution**: Distillation from Moonlight produced more distributed routing patterns (utilizing more experts) compared to regular fine-tuning, which maintained more concentrated expert usage.

- **Layer-Specific Effects**: Distillation shows stronger effects in middle layers, while regular fine-tuning affects later layers more prominently.

- **Consistency Across Tokens**: Distillation produces more consistent changes across different token types, whereas regular fine-tuning shows more token-type-specific changes.

## Visualization Highlights

### Routing Pattern Similarities

The analysis of routing pattern similarities across layers shows that distillation creates a distinctive signature that differs from regular fine-tuning:

- **Expert Selection Match Ratio**: The distilled model shows a distinctive pattern of expert selection compared to both the base model and the regularly fine-tuned model.

- **Positional Expert Matches**: The pattern of which expert positions (primary, secondary, etc.) are affected most by distillation provides a clear signal for detecting distilled models.

### Dimension-Level Changes

The dimension-level analysis revealed specific dimensions that are most affected during distillation:

- **Top Changed Dimensions**: Across layers, specific dimensions consistently showed larger changes in the distilled model compared to normal fine-tuning.

- **Dimensional Clustering**: The pattern of how dimensions cluster together during changes provides additional evidence of distillation versus regular fine-tuning.

### Token Embedding Analysis

The token embedding analysis showed:

- **Embedding Space Shifts**: Distillation causes systematic shifts in the embedding space that correlate with routing changes.

- **Token-Specific Effects**: The degree to which tokens are affected by distillation varies significantly, with some tokens maintaining similar representations and others changing dramatically.

## Conclusion

Our analysis demonstrates that routing patterns in MoE models can indeed be used to detect whether a model has been distilled from another model. The distillation process leaves distinctive fingerprints in:

1. The pattern of expert selection
2. The distribution of routing decisions
3. The specific dimensions that change most significantly
4. The correlation between embedding changes and routing changes

These findings provide a basis for developing methods to detect distilled models, which could be useful for:

- **Model Provenance**: Determining whether a model has been derived from another model
- **Distillation Quality**: Assessing how well the distillation process has transferred knowledge
- **Security Research**: Detecting potential unauthorized model copying or derivation

Future work could focus on quantifying these patterns more precisely and developing automated tools for distillation detection in MoE architectures.

## Appendix: Technical Details

### Model Architectures

All OLMoE models use a top-8 routing scheme with 64 experts per layer. The Moonlight model uses a different architecture with a top-6 routing scheme and shared experts, making direct comparison challenging. However, the patterns of change observed in the OLMoE model after distillation from Moonlight provide clear evidence of knowledge transfer.

### Methodology

Our analysis involved:

1. Dumping router token selection and hidden states from each model
2. Identifying common sequences processed by all models
3. Comparing expert selection patterns at the token level
4. Analyzing dimension-level changes in token embeddings
5. Correlating routing changes with embedding changes
6. Visualizing patterns to identify distinctive signatures of distillation

### Limitations

The analysis has several limitations:

- Limited to a specific pair of model architectures (OLMoE and Moonlight)
- Analysis focused on a subset of common sequences
- The distillation process used for these models may not be representative of all distillation approaches

Despite these limitations, the consistent patterns observed provide strong evidence that routing patterns can be used to detect distilled MoE models. 