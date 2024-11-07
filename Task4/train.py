import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
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

# Define layer-wise learning rates for transformer layers and task-specific heads
def get_layerwise_optimizer(model, base_lr=1e-5, head_lr=1e-4):
    """Returns an AdamW optimizer with different learning rates for different layers."""
    transformer_layers = list(model.transformer.encoder.layer)
    lr_multiplier = 0.9
    optimizer_grouped_parameters = []
    
    for i, layer in enumerate(transformer_layers):
        lr = base_lr * (lr_multiplier ** i)
        optimizer_grouped_parameters.append({
            'params': layer.parameters(),
            'lr': lr
        })
    
    optimizer_grouped_parameters += [
        {'params': model.classification_head_A.parameters(), 'lr': head_lr},
        {'params': model.classification_head_B.parameters(), 'lr': head_lr}
    ]
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    return optimizer

model = MultiTaskSentenceTransformer()
# Example of creating an optimizer with layer-wise learning rates
optimizer = get_layerwise_optimizer(model)

# Example custom Dataset class
class MultiTaskDataset(Dataset):
    def __init__(self, sentences, labels_A, labels_B, task):
        self.sentences = sentences
        self.labels_A = labels_A
        self.labels_B = labels_B
        self.task = task
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label_A = self.labels_A[idx]
        label_B = self.labels_B[idx]
        return sentence, label_A, label_B

# Training function
def train(model, train_dataset, optimizer, num_epochs=3, batch_size=8, task='A'):
    model.train()
    
    # Create DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Loss functions
    criterion_A = nn.CrossEntropyLoss()  # For sentence classification
    criterion_B = nn.CrossEntropyLoss()  # For sentiment analysis
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for batch in train_dataloader:
            sentences, labels_A, labels_B = batch
            optimizer.zero_grad()  # Clear gradients
            
            # Move data to device (e.g., GPU)
            sentences = sentences
            labels_A = labels_A
            labels_B = labels_B
            
            # Forward pass
            if task == 'A':
                logits = model(sentences, task='A')  # Sentence classification
                loss = criterion_A(logits, labels_A)
            elif task == 'B':
                logits = model(sentences, task='B')  # Sentiment analysis
                loss = criterion_B(logits, labels_B)
            else:
                raise ValueError(f"Unknown task: {task}")
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Print loss for the epoch
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_dataloader)}")

# Example usage
sentences = ["I love this product!", "I hate waiting in line.", "It's an average experience."]
labels_A = [0, 1, 2]  # Example labels for sentence classification (Task A)
labels_B = [2, 0, 1]  # Example labels for sentiment analysis (Task B)

# Create dataset
train_dataset = MultiTaskDataset(sentences, labels_A, labels_B, task='A')

# Train the model (example for Task A)
train(model, train_dataset, optimizer, num_epochs=3, batch_size=2, task='A')

# Train the model (example for Task B)
train(model, train_dataset, optimizer, num_epochs=3, batch_size=2, task='B')
