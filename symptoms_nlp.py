# %% [markdown]
# we will use nlp to convert symptoms to fit our dataset

# %%
import spacy as sp
import pandas as pd
import numpy as np


# %%
df1 = pd.read_csv("/Users/vedanta/Documents/VSCode/hack_x/Symptom-severity.csv")

# %%
df1['Symptom'] = df1['Symptom'].str.replace('_'," ")
df1

# %%

for i in df1:
    sentence = df1['Symptom']
sentence

    

# %%


# %%
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('distilbert-base-nli-mean-tokens')


sentence_embeddings = model.encode(sentence)

for sentence, embedding in zip(sentence, sentence_embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")



# %%
input_sentence = "runny nose"

# %%
input_sentence_embeding = model.encode(input_sentence)
print(input_sentence)
input_sentence_embeding

# %%
from sentence_transformers import SentenceTransformer, util
outputls = []
for i in sentence_embeddings:
    output =  util.pytorch_cos_sim(input_sentence_embeding,i)
    output = str(output)
    outputls.append(output)
outputls = [string.replace('tensor([[','').replace(']])','') for string in outputls]

outputls1 = []
for i in outputls:
    output = float(i)
    outputls1.append(output)

df1['embedded_scores'] = outputls1
df_sorted = df1.sort_values(by='embedded_scores', ascending=False)
symp = df_sorted.head(1)
symp

# %%
symptoms_arr = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

# %% [markdown]
# Creating threshold

# %%



