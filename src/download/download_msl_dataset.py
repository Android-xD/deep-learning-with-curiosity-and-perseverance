import requests
import os
import zipfile
from src.utils.configs import data_dir

url = r"https://zenodo.org/record/1049137/files/msl-images.zip?download=1"

req = requests.get(url)

print('Downloading Started')
zip_file = os.path.join(data_dir, "msl_images.zip")
out_dir = os.path.join(data_dir, "msl_images")

# Writing the file to the local file system
with open(zip_file, 'wb') as output_file:
    output_file.write(req.content)

print('Downloading Completed')

with zipfile.ZipFile(zip_file, 'r') as zip:
    zip.extractall(out_dir)
print('Extracting Completed')

# Delete the zip file
os.remove(zip_file)

print('Done')
