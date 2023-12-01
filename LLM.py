from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import os 
import openai
import matplotlib.pyplot as plt
from openai import OpenAI
import numpy as np
import csv
import nltk
import os
import pandas as pd
import sys
import re
#nltk.download('punkt')

#openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key="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"
client = OpenAI(api_key="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")
# chat=ChatOpenAI(openai_api_key="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")
# messages = [
#     SystemMessage(content="n."),
#     HumanMessage(content="What'?"),
# ]
# print(chat(messages))

def dataprocess():
  nparray=[]
  data = pd.read_csv("tweets.csv")
  file = open('tweets_cleaned.csv', mode='w', newline='',  encoding='utf-8')
  file2 = open('tweets_cleaned2.csv', mode='w', newline='',  encoding='utf-8')
  writer = csv.writer(file)
  writer2 = csv.writer(file2)
  for i in range(1000): #data.shape = (2811774, 7)
    pattern=r'@[\dA-Za-z]+'
    s=data.iloc[i]["text"]
    if not re.match(r'^\^', s):
      #nparray.append([re.sub(r'@\d+\s', '', data.iloc[i]["text"])])
      #writer.writerow(['Data', 'Label'] 
      writer.writerow([re.sub(pattern, '',s)])
    writer2.writerow([re.sub(pattern, '',s)])
  file.close()




def extract_sentences(file_path):
  with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()
    sentences = nltk.tokenize.sent_tokenize(text)
    sentences_array = np.array(sentences)
    file = open('belief_sentences.csv', mode='w', newline='',  encoding='utf-8')
    writer = csv.writer(file)
    for s in sentences_array:
      writer.writerow([s]) 
    return sentences_array
  #print(sentences_array[0:10])

def concat_data():
  data=[]
  formal_file = open('belief_sentences.csv', mode='r',  encoding='utf-8')
  casual_file = open('tweets_cleaned.csv', mode='r', encoding='utf-8')
  formal_reader = csv.reader(formal_file)
  casual_reader = csv.reader(casual_file)
  for i in range(100):
    try:
      # Now, next_line contains the values of the next line in the CSV file
      data.append((next(formal_reader),1))
      data.append((next(casual_reader),0))
    except StopIteration:
    # Handle the case when there are no more lines in the CSV file
      pass
  #print(data[0:10])
  if os.path.isfile("trueVSpredict_labels.npy"):
    trueVSpredict_labels = np.load("trueVSpredict_labels.npy").tolist()
  else:
    trueVSpredict_labels=[]
  for i in range(TEST):
    test_sentence=data[3+i]
    prompt=f"Help me classify some texts? Here are some examples: ({data[0]}, {data[1]}, {data[2]}). Now I have a new unlabeled sentence {test_sentence}. What do you think the label is? You can take a guess, but you need to give me an result! Label your answer as \"reasoning: \" and \"prediction: {{number}}\" "
    response = callLLM(prompt)
    response=response.choices[0].message.content
    # Use re.search() to check if the pattern is in the sentence
    match = re.search(r'prediction:\s*(\d+)', response)
    if match:
      prediction=int(match.group(1))
      trueVSpredict_labels.append([test_sentence[1],prediction])
    else:
      trueVSpredict_labels.append([test_sentence[1],-1])
    # print(f"prompt: {prompt}")
    # print(f"response: {response}")
    # print(f"trueVSpredict_labels: {trueVSpredict_labels}")
    # print("------------------")
    #print(f"test sentence {test_sentence}")
  np.save("trueVSpredict_labels.npy", np.array(trueVSpredict_labels))
  print(trueVSpredict_labels)
  print("--------------")
  return trueVSpredict_labels


def makefig(trueVSpredict_labels):
# Example array of pairs (assumed format)
  # Count correct and wrong predictions
  correct_predictions = sum(1 for a, b in trueVSpredict_labels if a == b)
  wrong_predictions = len(trueVSpredict_labels) - correct_predictions

  # Data for histogram
  categories = ['Correct Predictions', 'Wrong Predictions']
  values = [correct_predictions, wrong_predictions]
  plt.bar(categories, values, color=['green', 'grey'])
  plt.xlabel('Type of Prediction')
  plt.ylabel('Number of Sentences')
  plt.title('Accuracy of Prediction Given 3-Shot Prompting')
  plt.savefig("accuracy")







# def main():
#   pairs = [
#     ("BIG HOUSE", "false"),
#     ("brown cat","true"),
#     ("the cat is orange", "true")
#   ]
#   with open('pairs1.csv', 'w', newline='', encoding='utf-8') as file:
#       writer = csv.writer(file)
#       #writer.writerow(['Data', 'Label'] 
#       for pair in pairs:
#           writer.writerow(pair) #so in theory if the read is diff, the data structure is different! 


# def others():
#   data_label = []
#   # Open the CSV file for reading
#   with open('tweets.csv', 'r', encoding='utf-8') as file:
#     reader = csv.reader(file)
#     for row in reader:
#       pass
#   #print(data_label)

def callLLM(p):
  response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": p}]
  )
  return response 

if __name__ == "__main__":
    #dataprocess()
    #extract_sentences("belief.txt")
    #others()
    TEST=5


    trueVSpredict_labels=concat_data()
    makefig(trueVSpredict_labels)


# df = pandas.read_csv("data.csv",header=0, index_col=0)
# df.index.name = 'id'

