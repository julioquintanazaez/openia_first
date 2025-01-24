#https://colab.research.google.com/drive/1QLiVBR0X1ezECFDLaQSMLx1XY05WqwTW?usp=sharing#scrollTo=Dwas8OKFr5tx
#!pip install openai pandas requests google-colab
# Subimos 2 archivos CSV con nombre de archivo: products.csv que contiene el nombre de los productos y categories.csv que tiene las categorías que debe clasificar.
#from google.colab import files
#uploaded = files.upload()

import openai
import re
import pandas as pd
import time
import key_service 
import load_data


print(products_df.head())


def delayed_completion(delay_in_seconds: float = 1, **kwargs):
  """Delay a completion by a specified amount of time."""

  # Sleep para el retardo
  time.sleep(delay_in_seconds)

  # Llama a la API de finalización y devuelve el resultado
  return openai.Completion.create(**kwargs)

# Calcula el retraso en función de tu límite de tarifa
rate_limit_per_minute = 20
delay = 60.0 / rate_limit_per_minute


def get_prediction(products_number):
  for producto in products_df["product"][:products_number]:
    try:
      # -------------------------
      prompt = "Clasifica el siguiente producto de una tienda online de productos de canasta bàsica" + str(producto) + " en una de las siguientes categorías: \n " + category_list + "\n y devuelve únicamente el nombre de la categoría elegida"

      # Evitar la limitación de velocidad
      resp_producto = delayed_completion(
          delay_in_seconds=delay,
          model="text-davinci-003",
          prompt=prompt,
          max_tokens=1500,
          temperature=0.7,
      )

      # print(resp_producto)
      resp_producto = (resp_producto.choices[0].text)

      resp_producto.rstrip()
      resp_producto = resp_producto.replace('\n', '')
      resp_producto = resp_producto.replace('.', '')
      resp_producto = resp_producto.replace(':', '')
      result_df = result_df.append({"product": producto, "category": resp_producto}, ignore_index=True)

      return result_df
    except Exception as e:
      print(f"Error general: {str(e)}")


classification_df = get_prediction(50)

# Guardar el DataFrame en un archivo CSV
classification_df.to_csv("productos_clasificados.csv", index=False)

# Muestra el resultado
display(classification_df)

