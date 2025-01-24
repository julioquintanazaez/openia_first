import pandas as pd



# Subimos el CSV de productos a clasificar
df = pd.read_excel("kada_otros.xlsx")

products_df = df[["name", "tag"]].copy()


print(products_df["name"][0])