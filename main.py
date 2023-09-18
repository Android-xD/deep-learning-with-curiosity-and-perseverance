import os
from src.download.load_curiosity import load_curiosity
from src.download.load_perseverance import load_perceverance


# Create the data folder
data_dir = os.path.join(".", "data")

# Download the curiosity images
curiosity_dir = os.path.join(data_dir, "curiosity")
load_curiosity(curiosity_dir)

# Download the perseverance images
perseverance_dir = os.path.join(data_dir, "perseverance")
load_perceverance(perseverance_dir)