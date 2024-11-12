import pandas as pd
import os

from utils import json_to_df, get_most_common_word_simple

base_repo_dir = r"C:\Users\User\Documents\Ruppin\age_prediction_project_repo"  # replace for your own
train_path = os.path.join(base_repo_dir, "data_files", "train.json")

# extracting train data into a df:
train_df = json_to_df(json_file=train_path)
print(train_df.shape)
print(train_df.head((2)))


# Generate a basic feature DataFrame for the first 1000 posts
basic_feature_df = train_df[["age", "gender", "post"]].iloc[:1000]
# Ensure 'post_index' is correctly copied
basic_feature_df["post_index"] = basic_feature_df.index
# Calculate word count for each post
basic_feature_df["word_count"] = basic_feature_df["post"].apply(lambda x: len(x.split()) if pd.notnull(x) else 0)
# Find the most common word in each post
basic_feature_df["most_common_word"] = basic_feature_df["post"].apply(get_most_common_word_simple)

print(basic_feature_df.shape)
print(basic_feature_df.head())