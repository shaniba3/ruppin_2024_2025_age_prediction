from imports import *
from feature_functions import *

# הגדרת הנתיב הבסיסי לפרויקט
base_repo_dir = os.getcwd()

# קריאה לפונקציה לטעינת הנתונים
basic_feature_df = load_and_prepare_data(base_repo_dir)

# הצגת תוצאה ראשונית (לבדוק שהנתונים נטענו כהלכה)
print("Data loaded and prepared:")
print(basic_feature_df.head())

# שימוש בפונקציה להוספת הפיצ'ר stop_word_ratio
basic_feature_df = add_stop_word_ratio_feature(basic_feature_df)
# ניקוי מילות קישור והחלפת עמודת 'post'
basic_feature_df = remove_stop_words_from_posts(basic_feature_df)
# בדיקת ערכים חסרים בעמודת הגיל
check_missing_ages(basic_feature_df)
# הוספת עמודת המילה הנפוצה ביותר
basic_feature_df = add_most_common_word(basic_feature_df)
# הוספת עמודת מספר המילים
basic_feature_df = add_word_count(basic_feature_df)
# הוספת עמודת יחס מילים ייחודיות
basic_feature_df = add_unique_word_ratio(basic_feature_df)
# הוספת עמודת tfidf_score
basic_feature_df = add_tfidf_score(basic_feature_df)
# הוספת עמודת sentiment_score
basic_feature_df = add_sentiment_score(basic_feature_df)
# הוספת עמודת vader_sentiment_label
basic_feature_df = add_vader_sentiment_label(basic_feature_df)
# הוספת עמודת formality_score
basic_feature_df = add_formality_score(basic_feature_df)
# הוספת עמודת punctuation_ratio
basic_feature_df = add_punctuation_ratio(basic_feature_df)
# הוספת עמודת punctuation_correctness_score
basic_feature_df = add_punctuation_correctness_score(basic_feature_df)
# הוספת עמודת grammar_error_ratio
basic_feature_df = add_grammar_error_ratio(basic_feature_df)
# הוספת עמודת avg_word_length
basic_feature_df = add_avg_word_length(basic_feature_df)
# הוספת עמודות מנורמלות של סימני הפיסוק
basic_feature_df = add_normalized_punctuation_features(basic_feature_df)
# הוספת עמודות של חלוקת זמני הפעלים
basic_feature_df = add_verb_tense_distribution(basic_feature_df)

if __name__ == "__main__":
    # הקוד שלך שנטען את הנתונים
    print("Entering interactive mode...")
    import code
    code.interact(local=locals())
