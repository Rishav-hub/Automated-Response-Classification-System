# Automated-Response-Classification-System using NLP

## Problem Statement
After the rise in use of social media platforms, many people have started commenting their thoughts on these platform. Segregating positive and negative comments have become an important task.

## Solution Proposed
We will use different models like LSTM and GRU to train our model on the provided data.

## Approach

- The Data to be used is Sentiment140 dataset with 1.6 million tweets.

- The text processing would be done using torchtext or spacy. Reference.

- For feature engineering or Converting to embedding, we will use GLoVE or Word2Vec. Reference.

- The model to be used would be LSTM, GRU, and encode or decoder models.

- Compare different model performances.

## Use Case

The models that we are training can be inferenced on the Customer Response about the course which would reduce the work load of mentors and they can target only people with a negative response and solve their query.

## Tech Stack Used
1. Python 
2. FastAPI 
3. Pytorch
4. Docker
5. AWS
6. Azure

## Infrastructure required
1. AWS S3
2. Azure App service
3. Github Actions

## How to run

Step 1. Create a conda environment.
```
conda create -p env python=3.8 -y
```
```
conda activate ./env
````
Step 2. Install the requirements 
```
pip install -r requirements.txt
```
Step 3. Export the environment variable
```bash
export AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID>

export AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY>
```
Step 4. Run the application server
```
python app.py
```
Step 5. Train application
```bash
http://localhost:8000/train
```
Step 6. Prediction application
```bash
http://localhost:8000/predict
```
## Run locally

1. Check if the Dockerfile is available in the project directory

2. Build the Docker image
```
docker build -t sentiment .
```

3. Run the Docker image
```
docker run -d -p 8000:8000 -e AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID> -e AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY> sentiment
```

## Models Used
* Custom LSTM

## `sentiment` is the main package folder which contains 


**Components** : Contains all components of Deep Learning(CV) Project
- data_ingestion
- data_transformation
- model_training
- model_evaluation
- model_pusher
