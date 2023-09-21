import os
import requests
import json
import glob
import fnmatch
import src.utils.configs as configs
from src.utils.configs import data_dir
from tqdm import tqdm
from threading import Thread

secrets_file = open("../../secrets.json", "r")
secrets = json.load(secrets_file)
KEY = secrets["key"]

def load_curiosity_urls_of_sol(sol, url_dir, img_dir):
    url = f"https://api.nasa.gov/mars-photos/api/v1/rovers/curiosity/photos?sol={sol}&api_key={KEY}"
    os.makedirs(url_dir, exist_ok=True)
    json_filepath = os.path.join(url_dir, f"{sol}.json")
    if os.path.isfile(json_filepath):
        return

    response = requests.get(url)
    remaining_requests = response.headers._store['x-ratelimit-remaining'][1]
    print(f"number of remaining requests {remaining_requests}")
    if response.status_code == 200:
        data = response.json()
        photos = data.get('photos', [])
        urls = []
        file_paths = []

        for photo in photos:
            img_src = photo['img_src']
            camera_name = photo['camera']['name']
            file_name = os.path.basename(img_src)
            file_path = os.path.join(img_dir, camera_name, file_name)
            urls.append(img_src)
            file_paths.append(file_path)
        data = dict(zip(urls, file_paths))
        with open(json_filepath, "w") as f:
            json.dump(data, f)


def load_curiosity_urls():
    n=4000
    pbar = tqdm(total=n, desc="load_curiosity_urls")
    def task(sols):
        for sol in sols:
            if not os.path.isfile(os.path.join(data_dir, "urls", "curiosity", f"{sol}.json")):
                load_curiosity_urls_of_sol(sol, os.path.join(data_dir, "urls", "curiosity"), os.path.join(data_dir, "curiosity"))
            pbar.update(1)

    num_threads = 16
    threads = []
    for i in range(num_threads):
        sols = range(i, n, num_threads)
        thread = Thread(target= lambda : task(sols))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()


def load_perseverance_urls_of_page(page, url_dir, img_dir):
    url = "https://mars.nasa.gov/rss/api/"
    params = {
        "feed": "raw_images",
        "category": "mars2020,ingenuity",
        "feedtype": "json",
        "order": "sol desc",
        "ver": "1.2",
        "num": 100,
        "page":page
    }
    os.makedirs(url_dir, exist_ok=True)
    json_filepath = os.path.join(url_dir, f"{page}.json")
    if os.path.isfile(json_filepath):
        return

    response = requests.get(
                url, params={**params}
        )
    if response.status_code == 200:
        photos = response.json()["images"]
        urls = []
        file_paths = []

        for photo in photos:
            img_src = photo["image_files"]["full_res"]
            camera_name = photo['camera']['instrument']
            file_name = os.path.basename(img_src)
            file_path = os.path.join(img_dir, camera_name, file_name)
            urls.append(img_src)
            file_paths.append(file_path)
        data = dict(zip(urls, file_paths))
        with open(json_filepath, "w") as f:
            json.dump(data, f)


def load_perseverance_urls():
    n = 2872
    pbar = tqdm(total=n, desc="load_perseverance_urls")

    def task(pages):
        for page in pages:
            if not os.path.isfile(os.path.join(data_dir, "urls", "perseverance", f"{page}.json")):
                load_perseverance_urls_of_page(page, os.path.join(data_dir, "urls", "perseverance"), os.path.join(data_dir, "perseverance"))
            pbar.update(1)

    num_threads = 16
    threads = []
    for i in range(num_threads):
        pages = range(i, n, num_threads)
        thread = Thread(target= lambda : task(pages))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

def download_image(url, file_path, pbar):
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        pbar.update(1)
        return
    # create directory if necessary
    os.makedirs(os.path.dirname(file_path),exist_ok=True)

    response = requests.get(url)
    if response.status_code == 200:

        with open(file_path, "wb") as file:
            file.write(response.content)
            pbar.update(1)
    else:
        print("error")

def load_urls_from_json(rover = "curiosity"):
    url_dir = os.path.join(data_dir, "urls", rover, "*.json")
    json_filepaths = glob.glob(url_dir)
    data_dict = {}
    for json_filepath in json_filepaths:
        data_json = open(json_filepath, "r")
        try:
            data = json.load(data_json)
            data_dict.update(data)
        except:
            print(f"remove broken: {json_filepath}")
            os.remove(json_filepath)

    print(f"{len(data_dict)} urls found ")
    return data_dict

def load_images(image_data, pbar):
    for url, file_path in image_data.items():
        download_image(url, file_path, pbar)

def load_images_paralell(rover= "curiosity"):
    image_data = load_urls_from_json(rover)
    pbar = tqdm(total=len(image_data))
    urls, files = list(image_data.keys()), list(image_data.values())
    num_threads = 8
    threads = []
    for i in range(num_threads):
        thread = Thread(target= lambda : load_images(dict(zip(urls[i::num_threads], files[i::num_threads])),pbar))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

def load_selected_images_paralell(rover, path_reg):
    image_data = load_urls_from_json(rover)

    urls_files = []
    for url, file in image_data.items():
        if fnmatch.fnmatch(file, path_reg):
            urls_files.append((url, file))

    pbar = tqdm(total=len(urls_files))

    num_threads = 8
    threads = []
    for i in range(num_threads):
        thread = Thread(target= lambda : load_images(dict(urls_files[i::num_threads]), pbar))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

# load_curiosity_urls()
# load_selected_images_paralell("curiosity",configs.curiosity_mast_color_small) # 1.3 GB
# load_selected_images_paralell("curiosity",configs.curiosity_mast_color_small) # 0.9 GB
# load_perseverance_urls()
# load_selected_images_paralell("perseverance",configs.perseverance_mast_color) # 220 GB
# load_selected_images_paralell("perseverance",configs.perseverance_navcam_color) # 43 GB
