from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import os 
import openai
from openai import OpenAI
import numpy as np
import csv
import nltk
import pandas as pd
import re
#nltk.download('punkt')

openai.api_key = os.getenv("OPENAI_API_KEY")
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
  for i in range(1000): #data.shape = (2811774, 7)
    nparray.append([re.sub(r'@\d+\s', '', data.iloc[0]["text"])])
  
  with open('tweets_cleaned.csv', 'w', newline='', encoding='utf-8') as file:
      writer = csv.writer(file)
      #writer.writerow(['Data', 'Label'] 
      for s in nparray:
          writer.writerow(s)



def extract_sentences(file_path):
  with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()
    sentences = nltk.tokenize.sent_tokenize(text)
    sentences_array = np.array(sentences)
    return sentences_array
  #print(sentences_array[0:10])

def main():
  pairs = [
    ("BIG HOUSE", "false"),
    ("brown cat","true"),
    ("the cat is orange", "true")
  ]
  with open('pairs1.csv', 'w', newline='', encoding='utf-8') as file:
      writer = csv.writer(file)
      #writer.writerow(['Data', 'Label'] 
      for pair in pairs:
          writer.writerow(pair) #so in theory if the read is diff, the data structure is different! 


def others():
  data_label = []
  # Open the CSV file for reading
  with open('tweets.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    for i in range(5):
      print(reader[i])
  #print(data_label)

def callLLM():

  response = client.completions.create(
    model="text-davinci-003",
    # messages=[
    #   {"role": "system", "content": "You're a considerate person that ranks actions on a scale of most undesirable (-10) to most desirable (10). Give one score that you think fits the best."},
    #   #{"role": "user", "content": "Who won the world series in 2020?"},
    #   #{"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
    #   {"role": "user", "content": "What is the score of 'breaking a vase'? Just the number."}
    # ]
    prompt="You're a considerate person that adheres to social norms. Rank the action on a scale of -10 to 10, from most undesirable to most desirable."
  )
  print(response)

if __name__ == "__main__":
    dataprocess()
    #extract_sentences("belief.txt")
    #others()
# df = pandas.read_csv("data.csv",header=0, index_col=0)
# df.index.name = 'id'

