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



# Leer los datos
dataset = load_dataset("wikimedia/wikipedia", "20231101.en")
NUM_SAMPLES = 10000
articles = dataset["train"][:NUM_SAMPLES]["text"]
articles = [x.split("\n")[0] for x in articles]


print(len(articles))
print(articles[4])


# Instancia del agente IA y especifica el modelo a utilizar
llm = ChatOpenAI(model="gpt-3.5-turbo")
print(llm.model_name)


# Crear una plantilla de prompt 
prompt = ChatPromptTemplate.from_messages([
    ("system", """Your task is to assess the article and categorize the article into one of the following predfined categories:
                    'History', 'Geography', 'Science', 'Technology', 'Mathematics', 'Literature', 'Art', 'Music', 'Film', 'Television',
                    'Sports', 'Politics', 'Philosophy', 'Religion', 'Sociology', 'Psychology', 'Economics', 'Business', 'Medicine', 
                    'Biology', 'Chemistry', 'Physics', 'Astronomy', 'Environmental Science', 'Engineering', 'Computer Science', 'Linguistics',
                    'Anthropology', 'Archaeology', 'Education', 'Law', 'Military', 'Architecture', 'Fashion', 'Cuisine', 'Travel', 'Mythology',
                    'Folklore', 'Biography', 'Mythology', 'Social Issues', 'Human Rights', 'Technology Ethics', 'Climate Change', 'Conservation', 
                    'Urban Studies', 'Demographics', 'Journalism', 'Cryptocurrency', 'Artificial Intelligence'
                    you will output a json object containing the following information:

    {{"category": string}}"""), ("human", "{article}")
])


# Construir la cadena 
chain = prompt | llm


# Probar la predicción de una artículo
response = chain.invoke({"article": articles[2]})
print(response.response_metadata)


# Correr la inferencia del modelo 
results = []
for article in tqdm(articles[:100]):
    try:
        result = chain.invoke({"article": article})
        results.append(result)
    except Exception as e:
        print("Exception Occured", e)
        results.append("")


# Probar la predicción
print(articles[4], categories_list[4])


# Preproceso de la predicción
categories_list = [json.loads(x.content)["category"] for x in results]
ids = dataset["train"][:NUM_SAMPLES]["id"][:100]
print(len(ids), len(categories_list))
pd.DataFrame({
    "id": ids,
    "category": categories_list
})


# Estimación del costo
cost_df = pd.DataFrame([x.response_metadata["token_usage"] for x in results])
print(cost_df.shape)


input_tokens = cost_df["prompt_tokens"].sum()
output_tokens = cost_df["completion_tokens"].sum()
print(input_tokens, output_tokens)


costo = ((input_tokens * 1.5 ) / 10 ** 6) + ((output_tokens * 2 ) / 10 ** 6)
print(f"El costo de la redicción es: {costo}")
