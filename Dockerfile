# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt


# # Change directory to Task1 and run sentence_transformer.py file
# RUN cd Task1 && python sentence_transformer_model.py

# # Change directory to Task2 and run sentence_transformer.py file
# RUN cd Task2 && python multi_task_model.py



