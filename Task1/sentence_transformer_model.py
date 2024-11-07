import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class SentenceTransformerModel(nn.Module):
    '''
    1. Utilizing MiniLM model from 'sentence-transformers' for sentence embedding
    2. Using transformer library by Hugging Face for loading model
    '''
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        super(SentenceTransformerModel, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def mean_pooling(self, model_output, attention_mask):
        """Mean Pooling - Take attention mask into account for correct averaging."""
        token_embeddings = model_output[0]  # First element is the hidden states
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def forward(self, sentences):
        """Encode the input sentences into embeddings."""
        # Tokenize input sentences
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        
        # Get transformer output
        model_output = self.transformer(**encoded_input)
        
        # Perform mean pooling to get sentence embeddings
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embeddings


# Input examples
sentences = ['Hello World!', 'My name is Suraj.', "Today is a wonderful day, but the weather is not great."]

# Creating object of the model
model = SentenceTransformerModel()

# Generating embeddings
embed = model.forward(sentences)

print(f'Sentences embeddings: {embed}')
