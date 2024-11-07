# Task 1: Sentence Transformer Implementation

## Sentence Transformer Model

For this task, I’ll be leveraging a popular and efficient sentence transformer model, with a slight modification to generate fixed-length sentence embeddings.

## Key Design Decisions:
**Choice of Pretrained Model:** To prioritize both simplicity and performance, I will use the MiniLM model from Hugging Face. This lightweight model is fast while still providing competitive results.  
**Pooling Strategy:** To extract fixed-length sentence embeddings from the transformer output (which consists of token-level embeddings), I will incorporate a pooling mechanism. The standard pooling techniques are:  
1. **CLS Token Pooling:** Using the output from the [CLS] token.  
2. **Mean Pooling:** Calculating the mean of all token embeddings.  
I’ve chosen mean pooling, as it tends to better capture the overall meaning of a sentence compared to relying on just the [CLS] token.

## Key Components of the Code:
**Transformer Backbone:** The AutoModel from Hugging Face will be used to load a pretrained model (all-MiniLM-L6-v2), which is optimized for sentence-level tasks due to its small size and fast performance.  
**Tokenizer:** The AutoTokenizer handles the tokenization of input sentences, ensuring padding, truncation (with a default limit of 256 tokens), and conversion to tensors as required by the model.  
**Mean Pooling:** The mean_pooling function computes the mean of the token embeddings, accounting for the attention mask to exclude padding tokens, ensuring that only relevant tokens contribute to the final sentence embedding.

The model outputs fixed-length embeddings for each sentence. The length of the embeddings corresponds to the transformer’s hidden size (384 dimensions for all-MiniLM-L6-v2).

## Use Case

This model is suitable for encoding sentences and short paragraphs.

## Other Available Models

1. **Sentence BERT (SBERT)**  
2. **DistilBERT**  
3. **RoBERTa**  
4. **ALBERT**  
5. **BART**  

