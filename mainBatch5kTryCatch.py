from imports import *
from feature_functions import *
import gc
import psutil


# הגדרת הנתיב הבסיסי לפרויקט
base_repo_dir = os.getcwd()
def log_memory_usage():
    """מדפיס שימוש בזיכרון של התהליך"""
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1e6:.2f} MB")


# קריאה לפונקציה לטעינת הנתונים
posts_df, features_df = load_and_prepare_data(base_repo_dir)





# *** חישוב TF-IDF לכל הפוסטים (לפני חלוקת באצ'ים) ***
features_df = add_tfidf_score(posts_df, features_df)  # ציון TF-IDF לכל הפוסטים
export_table_to_parquet(features_df, "features_after_tfidf.parquet")

# *** עיבוד בבאצ'ים ***
batch_size = 5000
num_batches = (len(posts_df) // batch_size) + 1
for batch_num in range(num_batches):
    try:
        start_time = time.time()
        batch_file = f"test87K1_batch_{batch_num + 1}.parquet"

        if os.path.exists(batch_file):
            print(f"Batch {batch_num + 1} already processed. Skipping...")
            continue

        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(posts_df))
        print(f"Processing batch {batch_num + 1}/{num_batches} (rows {start_idx} to {end_idx})...")

        posts_batch = posts_df.iloc[start_idx:end_idx].copy()
        features_batch = features_df.iloc[start_idx:end_idx].copy()

        # פונקציות החישוב עם try/except
        features_batch = safe_function(add_stop_word_ratio, posts_batch, features_batch)
        check_missing_ages(features_batch)

        features_batch = safe_function(add_word_count, posts_batch, features_batch)
        features_batch = safe_function(add_unique_word_ratio, posts_batch, features_batch)
        features_batch = safe_function(add_avg_word_length, posts_batch, features_batch)

        features_batch = safe_function(add_sentiment_score, posts_batch, features_batch)
        features_batch = safe_function(add_vader_sentiment_score, posts_batch, features_batch)
        features_batch = safe_function(add_flair_sentiment_score, posts_batch, features_batch)
        features_batch = safe_function(add_bert_sentiment_score, posts_batch, features_batch)
        features_batch = safe_function(add_final_sentiment_score, None, features_batch)

        features_batch = safe_function(add_formality_score, posts_batch, features_batch)
        features_batch = safe_function(add_alternative_formality_score, posts_batch, features_batch)
        features_batch = safe_function(add_combined_formality_features, None, features_batch)
        features_batch = safe_function(add_punctuation_ratio, posts_batch, features_batch)
        features_batch = safe_function(add_normalized_punctuation_features, posts_batch, features_batch)
        features_batch = safe_function(add_writing_quality_score, posts_batch, features_batch)

        features_batch = safe_function(add_verb_tense_distribution, posts_batch, features_batch)
        features_batch = safe_function(add_verb_tense_distribution_alternative, posts_batch, features_batch)
        features_batch = safe_function(add_combined_tense_distribution, None, features_batch)

        export_table_to_parquet(features_batch, batch_file)

        del posts_batch, features_batch
        gc.collect()
        log_memory_usage()

        end_time = time.time()
        print(f"Finished processing batch {batch_num + 1}/{num_batches} in {end_time - start_time:.2f} seconds.")

    except Exception as e:
        print(f"Error processing batch {batch_num + 1}: {str(e)}")
        continue  # ממשיכים לבאצ' הבא במקום להתקע

# איחוד הטבלאות
print("Combining all batch results...")
all_features = pd.concat(
    [pd.read_parquet(f"test87K_batch_{i + 1}.parquet") for i in range(num_batches)],
    ignore_index=True
)

# ייצוא טבלה סופית
export_table_to_parquet(all_features, "final_features_table.parquet")
print("Final features table exported successfully to 'final_features_table1.parquet'.")

# *** פונקציות שלא בשימוש ***
# פונקציות אלו אינן מופעלות בקוד אך ייתכן ונשתמש בהן בהמשך.
# features_df = add_most_common_word(posts_df, features_df)  # מחשבת את המילה הנפוצה ביותר בכל פוסט
# features_df = add_grammar_error_ratio(posts_df, features_df)  # יחס שגיאות דקדוק לכל פוסט




# ייצוא טבלת הפוסטים
#posts_df.to_csv("posts_table100Ktest1.csv", index=False)
#print("Posts table exported successfully to 'posts_table100Ktest1.csv'.")







