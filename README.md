# ML Apprenticeship Take-Home

## Sentence Transformers and Multi-Task Learning

**Objective:** The goal of this exercise is to assess your ability to implement, train, and optimize neural network architectures, particularly focusing on transformers and multi-task learning extensions

## Task 1: Sentence Transformer Implementation
Implement a sentence transformer model using any deep learning framework of your choice. This model should be able to encode input sentences into fixed-length embeddings. Test your implementation with a few sample sentences and showcase the obtained embeddings. Describe any choices you had to make regarding the model architecture outside of the transformer backbone.

## Task 2: Multi-Task Learning Expansion
Expand the sentence transformer to handle a multi-task learning setting.
**Task A:** Sentence Classification – Classify sentences into predefined classes (you can make these up).
**Task B:** [Choose another relevant NLP task such as Named Entity Recognition, Sentiment Analysis, etc.] (you can make the labels up)
Describe the changes made to the architecture to support multi-task learning.

## Task 3: Training Considerations
Discuss the implications and advantages of each scenario and explain your rationale as to how the model should be trained given the following:
- If the entire network should be frozen.
- If only the transformer backbone should be frozen.
- If only one of the task-specific heads (either for Task A or Task B) should be frozen.
Consider a scenario where transfer learning can be beneficial. Explain how you would approach the transfer learning process, including:
- The choice of a pre-trained model.
- The layers you would freeze/unfreeze.
- The rationale behind these choices.

## Task 4: Layer-wise Learning Rate Implementation (BONUS)
- Implement layer-wise learning rates for the multi-task sentence transformer.
- Explain the rationale for the specific learning rates you've set for each layer.
- Describe the potential benefits of using layer-wise learning rates for training deep neural networks. Does the multi-task setting play into that?


## Note:

- I have utilized conda environment with python3.9
- Both python as well as Jupyter file is provided wherever found suited
- For Task4, the train.py file is not usable as I am not aware of need for training the model as it is not mentioned. I have just tried to provide a generalized code but in it pre-processing of dataset is not considered.
- Running the code
Buid docker image using
```
docker build -t fetch
```
Run the docker container 
```
docker run fetch mycontainer
```
Run interactive mode of docker
```
docker run -it fetch /bin/bash
``` 
Run the Task1: Sentence Transformer Implementation
```
cd /app/Task1
python sentence_transformer_model.py
```
Run the Task2: Multi-Task Learning Expansion

```
cd /app/Task2
python multi_task_model.py
```
- The current docker image size is 869 MB. Could have optimized it but avoiding due to time contraint.

