from imports import *
from feature_functions import *
# הגדרת הנתיב הבסיסי לפרויקט
base_repo_dir = os.getcwd()

# קריאה לפונקציה לטעינת הנתונים
posts_df, features_df = load_and_prepare_data(base_repo_dir)


# הצגת תוצאה ראשונית (לבדוק שהנתונים נטענו כהלכה)
print("Data loaded and prepared:")
print("Posts DataFrame:")
print(posts_df.head())
print("Features DataFrame:")
print(features_df.head())

# הוספת הפיצ'ר stop_word_ratio
features_df = add_stop_word_ratio(posts_df, features_df)
# עדכון טבלת הפוסטים לאחר הסרת מילות קישור
posts_df = remove_stop_words(posts_df)
# בדיקת ערכים חסרים בעמודת הגיל בטבלת הפיצ'רים
check_missing_ages(features_df)
# הוספת הפיצ'ר most_common_word
features_df = add_most_common_word(posts_df, features_df)
# הוספת הפיצ'ר word_count
features_df = add_word_count(posts_df, features_df)
# הוספת הפיצ'ר unique_word_ratio
features_df = add_unique_word_ratio(posts_df, features_df)
# הוספת הפיצ'ר tfidf_score
features_df = add_tfidf_score(posts_df, features_df)
# הוספת הפיצ'ר sentiment_score
features_df = add_sentiment_score(posts_df, features_df)
# הוספת הפיצ'ר vader_sentiment_label
features_df = add_vader_sentiment_label(posts_df, features_df)
# הוספת סנטימנט נוסף עם Flair
features_df = add_flair_sentiment_score(posts_df, features_df)
# הוספת סנטימנט נוסף עם BERT
#features_df = add_bert_sentiment_score(posts_df, features_df)
# הוספת הפיצ'ר formality_score
features_df = add_formality_score(posts_df, features_df)
# הוספת מדד פורמליות אלטרנטיבי (Gunning Fog)
features_df = add_alternative_formality_score(posts_df, features_df)
# הוספת הפיצ'ר punctuation_ratio
features_df = add_punctuation_ratio(posts_df, features_df)
# הוספת הפיצ'ר punctuation_correctness_score
features_df = add_punctuation_correctness_score(posts_df, features_df)
# הוספת הפיצ'ר grammar_error_ratio
features_df = add_grammar_error_ratio(posts_df, features_df)
# הוספת הפיצ'ר avg_word_length
features_df = add_avg_word_length(posts_df, features_df)
# הוספת הפיצ'רים המנורמלים של סימני הפיסוק
features_df = add_normalized_punctuation_features(posts_df, features_df)
# הוספת הפיצ'רים של חלוקת זמני הפעלים
features_df = add_verb_tense_distribution(posts_df, features_df)
# הוספת חלוקת זמני פעלים עם CoreNLP (Stanza)
features_df = add_verb_tense_distribution_alternative(posts_df, features_df)


# ייצוא טבלת הפוסטים
posts_df.to_csv("posts_table10Ktest1.csv", index=False)
print("Posts table exported successfully to 'posts_table10Ktest1.csv'.")

# ייצוא טבלת הפיצ'רים
features_df.to_csv("features_table10Ktest1.csv", index=False)
print("Features table exported successfully to 'features_table10Ktest1.csv'.")





