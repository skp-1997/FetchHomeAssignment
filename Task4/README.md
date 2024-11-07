# Task 4: Layer-wise Learning Rate Implementation
### Layer-Wise Learning Rates: Summary

Layer-wise learning rates is an advanced technique where different layers of a neural network, such as a transformer-based model, are assigned different learning rates. This is especially useful for models with a pre-trained backbone (e.g., MiniLM, BERT) and task-specific heads.

#### Rationale:
- **Lower layers**: Capture general knowledge (e.g., token embeddings, sentence structure).
- **Higher layers**: Capture task-specific features.
- In **multi-task learning** or **transfer learning**, higher layers are fine-tuned more aggressively, while lower layers are adjusted more cautiously.

#### Implementation:
- **Optimizer**: Use AdamW with parameter groups for different learning rates.
- **Base Learning Rate for Backbone**: A low learning rate is applied to avoid drastic changes to pre-trained knowledge.
- **Higher Learning Rates for Task-Specific Heads**: Task-specific heads (e.g., classification heads) receive a higher learning rate for faster adaptation.
- **Layer-Wise Decay**: Apply a decay factor (e.g., 0.9) to reduce learning rates for lower layers.

#### Benefits:
- **Better Fine-Tuning**: Higher layers adapt quickly to new tasks, while lower layers retain general knowledge.
- **Efficient Training**: Focuses fine-tuning on layers that need more adaptation, reducing unnecessary adjustments to lower layers.
- **Reduces Catastrophic Forgetting**: Preserves pre-trained knowledge by adjusting the backbone more cautiously.
- **Multi-Task Learning**: Helps maintain balance by preventing drastic changes to the shared backbone while allowing task-specific heads to specialize.

This approach optimizes both the model's efficiency and its ability to retain pre-trained knowledge.