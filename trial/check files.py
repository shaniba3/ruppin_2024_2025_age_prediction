import pandas as pd
import json

# נתיב מלא לקובץ
file_path = 'C:/Users/Tomer/PycharmProjects/ruppin_2024_2025_age_prediction/data/train_100000.json'

# טעינת הנתונים מהקובץ JSON
with open(file_path, 'r') as file:
    data = json.load(file)

# המרת הנתונים לDataFrame
df = pd.DataFrame(data)

# הדפסת ה-shape של ה-DataFrame
print("Shape of the DataFrame:", df.shape)
print(df.head() )