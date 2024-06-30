#!/usr/bin/env python3.12

# https://ollama.com/blog/embedding-models

import os
import sys
import json
import numpy as np

import ollama
from ollama import Client
import faiss

MODEL_AI = "llama3"
MODEL_EMBEDDINGS = "mxbai-embed-large"


def distance_l2(v1: list, v2: list) -> float:
    return np.linalg.norm(np.array(v2) - np.array(v1))


def cosine_angle(v1: list, v2: list) -> float:
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


documents = [
  "Llamas are members of the camelid family meaning they're pretty closely related to vicuñas and camels.",
  "Llamas were first domesticated and used as pack animals 4,000 to 5,000 years ago in the Peruvian highlands.",
  "Llamas can grow as much as 6 feet tall though the average llama between 5 feet 6 inches and 5 feet 9 inches tall.",
  "Llamas weigh between 280 and 450 pounds and can carry 25 to 30 percent of their body weight.",
  "Llamas are vegetarians and have very efficient digestive systems.",
  "Llamas live to be about 20 years old, though some only live for 15 years and others live to be 30 years old.",
]

embeddings = []

# store each document in a vector embedding database
for i, d in enumerate(documents):
    response = ollama.embeddings(model=MODEL_EMBEDDINGS, prompt=d)
    if len(embeddings) == 0:
        embeddings = np.array(response["embedding"], dtype=np.float32).reshape(1, -1)
    else:
        embeddings = np.vstack([embeddings, response["embedding"]])

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

# an example prompt
query = "What animals are llamas related to?"
print(f"query    > {query:s}")

# generate an embedding for the prompt and retrieve the most relevant doc
embeddings_query = np.array(ollama.embeddings(model=MODEL_EMBEDDINGS, prompt=query)["embedding"], dtype=np.float32).reshape(1, -1)

D, I = index.search(embeddings_query, 1)

context = documents[I[0][0]]

print(f"response > {context:s}")

