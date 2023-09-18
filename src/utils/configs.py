import os

data_dir = os.path.join("..", "..", "data")
log_dir = os.path.join("..", "..", "logs")
trained_weights_dir = os.path.join("..", "..", "trained_weights")

# define the type of images we use from curiosity
curiosity_mast_color_large = os.path.join(data_dir, "curiosity", "MAST", "*E1_DXXX.jpg")
curiosity_mast_color_small = os.path.join(data_dir, "curiosity", "MAST", "*I1_DXXX.jpg")

curiosity_navcam_color_large = os.path.join(data_dir, "curiosity", "NAVCAM_*", "N*A_*EDR_F*.png")
curiosity_navcam_color_small = os.path.join(data_dir, "data", "curiosity", "NAVCAM_*", "N*A_*EDR_D*.png")

# define the type of images we use from perceverance
perseverance_mast_color = os.path.join(data_dir, "perseverance", "MCZ_*", "ZR0*EBY*.png")
#perseverance_mast_bayer = os.path.join(data_dir, "perseverance", "MCZ_*", "ZR0*ECM*.png")

perseverance_navcam_color = os.path.join(data_dir, "perseverance", "NAVCAM_*", "N*F_*.png")
perseverance_navcam_gray_B = os.path.join(data_dir, "perseverance", "NAVCAM_*", "N*B_*.png")
perseverance_navcam_gray_G = os.path.join(data_dir, "perseverance", "NAVCAM_*", "N*G_*.png")

