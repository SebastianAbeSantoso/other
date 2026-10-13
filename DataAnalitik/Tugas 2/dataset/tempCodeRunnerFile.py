try:
    fire = pd.read_csv("fire.csv")
except FileNotFoundError:
    fire = pd.read_csv("fire_combined.csv")
