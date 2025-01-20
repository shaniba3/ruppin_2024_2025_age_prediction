from imports import *
from feature_functions import *

# הגדרת הנתיב הבסיסי לפרויקט
base_repo_dir = os.getcwd()

# קריאה לפונקציה לטעינת הנתונים
posts_df, features_df = load_and_prepare_data(base_repo_dir)





# *** חישוב TF-IDF לכל הפוסטים (לפני חלוקת באצ'ים) ***
features_df = add_tfidf_score(posts_df, features_df)  # ציון TF-IDF לכל הפוסטים
export_table_to_parquet(features_df, "features_after_tfidf.parquet")

# *** עיבוד בבאצ'ים ***
batch_size = 10000
num_batches = (len(posts_df) // batch_size) + 1
for batch_num in range(num_batches):
    start_time = time.time()  # שמירת זמן התחלת הבאץ'
    batch_file = f"test100K_batch_{batch_num + 1}.parquet"  # שם קובץ הבאץ'

    # בדיקה אם הקובץ כבר קיים
    if os.path.exists(batch_file):
        print(f"Batch {batch_num + 1} already processed. Skipping...")
        continue

    start_idx = batch_num * batch_size
    end_idx = min(start_idx + batch_size, len(posts_df))
    print(f"Processing batch {batch_num + 1}/{num_batches} (rows {start_idx} to {end_idx})...")

    # יצירת באץ'
    posts_batch = posts_df.iloc[start_idx:end_idx].copy()
    features_batch = features_df.iloc[start_idx:end_idx].copy()

    # *** קבוצה 1: הכנה בסיסית של הנתונים ***
    features_batch = add_stop_word_ratio(posts_batch, features_batch)  # יחס מילות קישור
    posts_batch = remove_stop_words(posts_batch)  # הסרת מילות קישור
    check_missing_ages(features_batch)  # בדיקה אם יש ערכים חסרים

    # *** קבוצה 2: סטטיסטיקות בסיסיות של טקסט ***
    features_batch = add_word_count(posts_batch, features_batch)  # ספירת מילים
    features_batch = add_unique_word_ratio(posts_batch, features_batch)  # יחס מילים ייחודיות
    features_batch = add_avg_word_length(posts_batch, features_batch)  # אורך ממוצע של מילים

    # *** קבוצה 3: סנטימנט ***
    features_batch = add_sentiment_score(posts_batch, features_batch)  # TextBlob
    features_batch = add_vader_sentiment_score(posts_batch, features_batch)  # Vader
    features_batch = add_flair_sentiment_score(posts_batch, features_batch)  # Flair
    features_batch = add_bert_sentiment_score(posts_batch, features_batch)  # BERT
    features_batch = add_final_sentiment_score(features_batch)  # "הרוב קובע"

    # *** קבוצה 4: פורמליות ואיכות כתיבה ***
    features_batch = add_formality_score(posts_batch, features_batch)  # פורמליות (Flesch)
    features_batch = add_alternative_formality_score(posts_batch, features_batch)  # פורמליות (Gunning Fog)
    features_batch = add_combined_formality_features(features_batch)  # שילוב מדדי פורמליות
    features_batch = add_punctuation_ratio(posts_batch, features_batch)  # יחס סימני פיסוק
    features_batch = add_normalized_punctuation_features(posts_batch, features_batch)  # יחס מנורמל
    features_batch = add_writing_quality_score(posts_batch, features_batch)  # איכות כתיבה

    # *** קבוצה 5: חלוקת זמני פעלים ***
    features_batch = add_verb_tense_distribution(posts_batch, features_batch)  # NLTK
    features_batch = add_verb_tense_distribution_alternative(posts_batch, features_batch)  # Stanza
    features_batch = add_combined_tense_distribution(features_batch)  # ממוצע בין חלוקת הזמנים

    # ייצוא טבלה מעודכנת לכל באץ'
    export_table_to_parquet(features_batch, batch_file)

    # מדידת זמן וסיום
    end_time = time.time()  # זמן סיום הבאץ'
    batch_duration = end_time - start_time  # זמן שעבר
    print(f"Finished processing batch {batch_num + 1}/{num_batches} in {batch_duration:.2f} seconds.")

# איחוד הטבלאות
print("Combining all batch results...")
all_features = pd.concat(
    [pd.read_parquet(f"test100K_batch_{i + 1}.parquet") for i in range(num_batches)],
    ignore_index=True
)

# ייצוא טבלה סופית
export_table_to_parquet(all_features, "final_features_table.parquet")
print("Final features table exported successfully to 'final_features_table.parquet'.")

# *** פונקציות שלא בשימוש ***
# פונקציות אלו אינן מופעלות בקוד אך ייתכן ונשתמש בהן בהמשך.
# features_df = add_most_common_word(posts_df, features_df)  # מחשבת את המילה הנפוצה ביותר בכל פוסט
# features_df = add_grammar_error_ratio(posts_df, features_df)  # יחס שגיאות דקדוק לכל פוסט




# ייצוא טבלת הפוסטים
#posts_df.to_csv("posts_table100Ktest1.csv", index=False)
#print("Posts table exported successfully to 'posts_table100Ktest1.csv'.")







