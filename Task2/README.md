# Task 2: Multi-Task Learning Expansion

## Design Choices Explanation:
Pretrained Model Selection:

I chose the all-MiniLM-L6-v2 model because it strikes a good balance between speed and accuracy for sentence embeddings. It is efficient for downstream tasks like sentence similarity and clustering.
Pooling Strategy:

Mean Pooling was chosen over the [CLS] token pooling since it tends to generalize better for most sentence embedding tasks by capturing information from all tokens, not just the first one.
Padding & Truncation:

I enabled padding to ensure that all input sentences are padded to the same length, making it suitable for batch processing. Truncation ensures that sentences longer than the model's maximum input length are shortened appropriately.
No Additional Layers:

Outside of the transformer backbone and pooling layer, no additional layers were added. The choice of directly using the embeddings without further transformation aligns with common usage in sentence-transformer models.