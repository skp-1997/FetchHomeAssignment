import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# Define the model class
class MultiTaskSentenceTransformer(nn.Module):
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', num_classes_A=3, num_classes_B=3):
        super(MultiTaskSentenceTransformer, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Task A: Sentence Classification Head
        self.classification_head_A = nn.Linear(self.transformer.config.hidden_size, num_classes_A)
        
        # Task B: Sentiment Analysis Head
        self.classification_head_B = nn.Linear(self.transformer.config.hidden_size, num_classes_B)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def forward(self, sentences, task='A'):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        model_output = self.transformer(**encoded_input)
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        
        if task == 'A':
            return self.classification_head_A(sentence_embeddings)
        elif task == 'B':
            return self.classification_head_B(sentence_embeddings)

# Instantiate the model
model = MultiTaskSentenceTransformer()

# Define layer-wise learning rates for transformer layers and task-specific heads
def get_layerwise_optimizer(model, base_lr=1e-5, head_lr=1e-4):
    """Returns an AdamW optimizer with different learning rates for different layers."""
    # Extract the layers from the transformer (BERT/MiniLM etc.)
    transformer_layers = list(model.transformer.encoder.layer)
    
    # Define learning rate decay: lower layers get smaller learning rates
    lr_multiplier = 0.9
    optimizer_grouped_parameters = []
    
    # Assign lower learning rates to lower layers of the transformer
    for i, layer in enumerate(transformer_layers):
        lr = base_lr * (lr_multiplier ** i)
        optimizer_grouped_parameters.append({
            'params': layer.parameters(),
            'lr': lr
        })
    
    # Assign higher learning rates for the task-specific heads
    optimizer_grouped_parameters += [
        {'params': model.classification_head_A.parameters(), 'lr': head_lr},
        {'params': model.classification_head_B.parameters(), 'lr': head_lr}
    ]
    
    # Use AdamW optimizer
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    return optimizer

# Example of creating an optimizer with layer-wise learning rates
optimizer = get_layerwise_optimizer(model)
# This can be used for training by defining training function given pre-processed training data is available
