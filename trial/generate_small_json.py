import pandas as pd
import os
import random
import json

from utils import json_to_list

base_repo_dir = r"C:\Users\User\Documents\Ruppin\age_prediction_project_repo"  # replace for your own
train_path = os.path.join(base_repo_dir, "data", "train.json")

train_data = json_to_list(train_path)

print(len(train_data))

smaller_train_data = random.sample(train_data, 1000)
print(len(smaller_train_data))

smaller_train_data_path = os.path.join(base_repo_dir, "data", "train_1000.json")
with open(smaller_train_data_path, "w") as fp:
    json.dump(smaller_train_data, fp)

small_train_data_2 = json_to_list(smaller_train_data_path)

