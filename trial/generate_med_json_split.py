import os
import json
from utils import json_to_dict  # ודא שהייבוא נכון

# נתיב הבסיס של הפרויקט
base_repo_dir = r"C:\Users\Tomer\PycharmProjects\ruppin_2024_2025_age_prediction"

# נתיב הקובץ המקורי
train_path = os.path.join(base_repo_dir, "data_files", "train.json")

# טוען את כל הנתונים מקובץ המקור
train_data = json_to_dict(train_path)
print("Total records in train.json:", len(train_data))

# מספר קבוצות לחלוקה
num_splits = 6
split_size = len(train_data) // num_splits  # גודל כל קבוצה
print(f"Each group will have approximately {split_size} records.")

# יצירת התיקייה לפיצול הקבצים
output_dir = os.path.join(base_repo_dir, "data", "split_files")
os.makedirs(output_dir, exist_ok=True)

# חלוקת הנתונים ושמירה לקבצים
for i in range(num_splits):
    start_idx = i * split_size
    if i == num_splits - 1:  # הקבוצה האחרונה תקבל את כל הרשומות שנותרו
        end_idx = len(train_data)
    else:
        end_idx = (i + 1) * split_size

    split_data = train_data[start_idx:end_idx]
    output_path = os.path.join(output_dir, f"train_split_{i + 1}.json")

    with open(output_path, "w") as fp:
        json.dump(split_data, fp)
    print(f"Created {output_path} with {len(split_data)} records.")

print("Data successfully split into 6 groups.")

