import vertexai
from vertexai.preview.language_models import TextGenerationModel

import queue
import threading
import time
import os

project_id = "rntbci-digital-innovation-lab"
os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
from google.cloud import aiplatform

aiplatform.init()

import numpy as np
import pandas as pd
# from sklearn.metrics import accuracy_score, confusion_matrix

def predict_large_language_model(
    model_name: str,
    temperature: float,
    max_decode_steps: int,
    top_p: float,
    top_k: int,
    content: str,
    tuned_model_name: str = "",
    ) :
    """Predict using a Large Language Model."""
    
    model = TextGenerationModel.from_pretrained(model_name)
    if tuned_model_name:
      model = model.get_tuned_model(tuned_model_name)
    response = model.predict(
        content,
        temperature=temperature,
        max_output_tokens=max_decode_steps,
        top_k=top_k,
        top_p=top_p,)
    return response.text

# def classify_review(review):
#     content = prompt.format(review=review)
#     response_text = predict_large_language_model(
#         "text-bison@001", 
#         temperature=0.2, 
#         max_decode_steps=5, 
#         top_p=0.8, 
#         top_k=1, 
#         content=content)
    # if response_text.lower() == 'negative':
    #     return 0
    # elif response_text.lower() == 'positive':
    #     return 1
    # else:
    #     return 2
    

prompt3 = '''explain the below python code line by line
code:  
read csv file
check info
explanation:
'''
# Classify the sentiment of the message: negative

# input: This Charles outing is decent but this is a pretty low-key performance. Marlon Brando stands out. There\'s a subplot with Mira Sorvino and Donald Sutherland that forgets to develop and it hurts the film a little. I\'m still trying to figure out why Charlie want to change his name.
# Classify the sentiment of the message: negative

# input: My family has watched Arthur Bach stumble and stammer since the movie first came out. We have most lines memorized. I watched it two weeks ago and still get tickled at the simple humor and view-at-life that Dudley Moore portrays. Liza Minelli did a wonderful job as the side kick - though I\'m not her biggest fan. This movie makes me just enjoy watching movies. My favorite scene is when Arthur is visiting his fiancée\'s house. His conversation with the butler and Susan\'s father is side-spitting. The line from the butler, "Would you care to wait in the Library" followed by Arthur\'s reply, "Yes I would, the bathroom is out of the question", is my NEWMAIL notification on my computer.
# Classify the sentiment of the message: positive

# input: {review}
# Classify the sentiment of the message: 
# '''

# review = "Something surprised me about this movie - it was actually original. It was not the same old recycled crap that comes out of Hollywood every month. I saw this movie on video because I did not even know about it before I saw it at my local video store. If you see this movie available - rent it - you will not regret it."
# content = prompt.format(review=review)
prompt2 = ''' you are sentiment predictor bot. find the sentiment from the ginve  input.if there are multiple sentiments identified in the given input provide the overall sentiment.
input: I like the negative comments of the movie and it is amazing.
Entities: Restaurant location, food, parking 

input: {review}
Entities:
'''



prompt1 = ''' Predict the exact sentiment of the given input.
input: {review}
Sentiment:
'''
review = "I like the negative comments of the movie and the negative roles of the movie."
content = prompt3




response_text = predict_large_language_model(
    "text-bison@001", 
    temperature=0.2, 
    max_decode_steps=5, 
    top_p=0.8, 
    top_k=4, 
    content=content)
print(response_text)
    