import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class MultiTaskSentenceTransformer(nn.Module):
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', num_classes_A=3, num_classes_B=3):
        super(MultiTaskSentenceTransformer, self).__init__()
        # Shared Transformer Encoder
        self.transformer = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Task A: Sentence Classification Head
        self.classification_head_A = nn.Linear(self.transformer.config.hidden_size, num_classes_A)
        
        # Task B: Sentiment Analysis Head
        self.classification_head_B = nn.Linear(self.transformer.config.hidden_size, num_classes_B)
        
        # Task A labels (mapping index to sentence classification label)
        self.sentence_classification_labels = {0: 'class_1', 1: 'class_2', 2: 'class_3'}
        
        # Sentiment labels (mapping index to sentiment type)
        self.sentiment_labels = {0: 'negative', 1: 'neutral', 2: 'positive'}

    def mean_pooling(self, model_output, attention_mask):
        """Mean Pooling - Take attention mask into account for correct averaging."""
        token_embeddings = model_output[0]  # First element is the hidden states
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def forward(self, sentences, task='A'):
        """Forward pass with multi-task support."""
        # Tokenize input sentences
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        
        # Get transformer output
        model_output = self.transformer(**encoded_input)
        
        # Perform mean pooling to get sentence embeddings
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        
        # Route to the task-specific head
        if task == 'A':  # Task A: Sentence Classification
            logits = self.classification_head_A(sentence_embeddings)
            probs = F.softmax(logits, dim=-1)  # Apply softmax for sentence classification output
            
            # Convert logits to sentence classification labels (predictions)
            sentence_predictions = torch.argmax(probs, dim=-1)  # Get the index of the highest probability class
            sentence_labels = [self.sentence_classification_labels[idx.item()] for idx in sentence_predictions]
            
            return sentence_labels, probs  # Return both labels and probabilities

        elif task == 'B':  # Task B: Sentiment Analysis
            logits = self.classification_head_B(sentence_embeddings)
            probs = F.softmax(logits, dim=-1)  # Apply softmax for sentiment analysis output
            
            # Convert logits to sentiment labels (predictions)
            sentiment_predictions = torch.argmax(probs, dim=-1)  # Get the index of the highest probability class
            sentiment_labels = [self.sentiment_labels[idx.item()] for idx in sentiment_predictions]
            
            return sentiment_labels, probs  # Return both labels and probabilities
        
        else:
            raise ValueError(f"Unknown task: {task}")

# Example usage
model = MultiTaskSentenceTransformer()

# Example sentences for sentence classification and sentiment analysis
sentences = ["I love this product!", "I hate waiting in line.", "It's an average experience."]

# Forward pass for Task A (Sentence Classification)
sentence_labels, sentence_probs = model(sentences, task='A')
print("Sentence Classification Results:")
for sentence, label, prob in zip(sentences, sentence_labels, sentence_probs):
    print(f"Sentence: {sentence}")
    print(f"Predicted Label: {label}")
    print(f"Probabilities: {prob}\n")

# Forward pass for Task B (Sentiment Analysis)
sentiment_labels, sentiment_probs = model(sentences, task='B')
print("Sentiment Analysis Results:")
for sentence, label, prob in zip(sentences, sentiment_labels, sentiment_probs):
    print(f"Sentence: {sentence}")
    print(f"Predicted Sentiment: {label}")
    print(f"Probabilities: {prob}\n")
