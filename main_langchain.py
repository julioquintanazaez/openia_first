#https://github.com/srinathmkce/TheAIGuy/blob/main/NLP/experiments/classification/Experiment-1%20GPT3.5-Part2.ipynb
import json
import os
#import getpass
import pandas as pd
from datasets import Dataset, load_dataset
from tqdm import tqdm
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import key_service
import load_data


print(products_df.head())


# Instancia del agente IA y especifica el modelo a utilizar
llm = ChatOpenAI(model="gpt-3.5-turbo")
print(llm.model_name)


tag_categories = products_df["tag"].unique()

# Crear una plantilla de prompt 
prompt = ChatPromptTemplate.from_messages([
    ("system", """Your task is to assess the product and categorize the product into
     one of the following predfined categories:
     {tag_categories}
    {{"category": string}}""")       # ("human", "{article}")
])


# Construir la cadena 
chain = prompt | llm


# Probar la predicción de una artículo
response = chain.invoke({"product": products_df["name"][0]})
print(response.response_metadata)


# Correr la inferencia del modelo 
def get_predictions(N):
    results = []
    for product in tqdm(products_df["name"][:N]):
        try:
            result = chain.invoke({"product": product})
            results.append(result)
        except Exception as e:
            print("Exception Occured", e)
            results.append("")

    return results


# Get predictios from the model
categories_list = get_predictions(50)

# Probar la predicción
print(products_df["name"][:0], categories_list[4])


