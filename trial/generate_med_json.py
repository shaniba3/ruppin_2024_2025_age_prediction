import os
import json
import random
from utils import json_to_dict  # ודא שהייבוא נכון

base_repo_dir = r"C:\Users\Tomer\PycharmProjects\ruppin_2024_2025_age_prediction"
train_path = os.path.join(base_repo_dir, "data_files", "train.json")

# טוען את כל הנתונים מקובץ המקור
train_data = json_to_dict(train_path)
print("Total records in train.json:", len(train_data))

# דוגם 10,000 רשומות
smaller_train_data_100000 = random.sample(train_data, k=100000)

# שומר את הקובץ החדש
smaller_train_data_100000_path = os.path.join(base_repo_dir, "data", "train_100000.json")
with open(smaller_train_data_100000_path, "w") as fp:
    json.dump(smaller_train_data_100000, fp)
print("Created train_10000.json with 100,000 records.")
