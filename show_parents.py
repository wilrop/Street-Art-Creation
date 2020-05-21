import os
from PIL import Image
import pandas as pd
from visualisation import plot_gen_parent


df = pd.read_csv("D:/Documenten/Graffiti-Gan-Results/artworks.csv")

for idx, row in df.iterrows():
    print(idx)
    gen_path = row["Path"]
    parent_path = row["Parent Path"]
    sim = row["Similarity"]
    generated = Image.open(gen_path)
    parent = Image.open(parent_path)
    if parent.mode != 'RGB':
        parent = parent.convert('RGB')
    plot_gen_parent(generated, parent, sim)

