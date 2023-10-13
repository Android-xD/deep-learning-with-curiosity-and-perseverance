import os
import json

# read the username from the secrets file
secrets_file = open("../../secrets.json", "r")
secrets = json.load(secrets_file)
user = secrets["email"][:7]

data_dir = os.path.join("/cluster", "scratch", user, "mars", "data")
if not os.path.isdir(data_dir):
    data_dir = os.path.join("..", "..", "data")

log_dir = os.path.join("..", "..", "logs")
fig_dir = os.path.join("..", "..", "figures")
trained_weights_dir = os.path.join("..", "..", "trained_weights")

# define the type of images we use from curiosity
curiosity_mast_color_large = os.path.join(data_dir, "curiosity", "MAST", "*E1_DXXX.jpg")
curiosity_mast_color_small = os.path.join(data_dir, "curiosity", "MAST", "*I1_DXXX.jpg")
curiosity_navcam_gray = os.path.join(data_dir, "curiosity", "NAVCAM", "N?A_*.JPG")


# define the type of images we use from perceverance
perseverance_mast_color = os.path.join(data_dir, "perseverance", "MCZ_*", "Z*0*EBY*.png")
#perseverance_mast_bayer = os.path.join(data_dir, "perseverance", "MCZ_*", "Z*0*ECM*.png")

perseverance_navcam_color = os.path.join(data_dir, "perseverance", "NAVCAM_*", "N?F_*.png")
perseverance_navcam_gray_B = os.path.join(data_dir, "perseverance", "NAVCAM_*", "N?B_*.png")
perseverance_navcam_gray_G = os.path.join(data_dir, "perseverance", "NAVCAM_*", "N?G_*.png")

# define the type of images we use from msl dataset
msl_color = os.path.join(data_dir, "msl_images", "calibrated", "*JPG")

# define the type of images we use from msl dataset
msl_train_file = os.path.join(data_dir, "msl_images", "train-calibrated-shuffled.txt")
msl_test_file = os.path.join(data_dir, "msl_images", "test-calibrated-shuffled.txt")
msl_val_file = os.path.join(data_dir, "msl_images", "val-calibrated-shuffled.txt")

image_types = {
    "curiosity_mast_color_small": curiosity_mast_color_small,
    "curiosity_mast_color_large": curiosity_mast_color_large,
    "perseverance_mast_color": perseverance_mast_color,
    "perseverance_navcam_color": perseverance_navcam_color
}