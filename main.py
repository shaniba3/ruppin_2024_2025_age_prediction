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

# *** קבוצה 1: הכנה בסיסית של הנתונים ***
# בקבוצה זו אנו מבצעים פעולות ראשוניות כמו חישוב יחס מילות קישור והסרתן.
features_df = add_stop_word_ratio(posts_df, features_df)  # חישוב יחס מילות קישור (Stop Words) עבור כל פוסט
posts_df = remove_stop_words(posts_df)  # הסרת מילות קישור מהטקסטים
check_missing_ages(features_df)  # בדיקה אם יש ערכים חסרים בעמודת הגיל

# *** קבוצה 2: סטטיסטיקות בסיסיות של טקסט ***
# כאן מחשבים נתונים כמו אורך הפוסט, יחס מילים ייחודיות, ועוד מאפיינים בסיסיים של טקסטים.
features_df = add_word_count(posts_df, features_df)  # ספירת המילים בכל פוסט
features_df = add_unique_word_ratio(posts_df, features_df)  # חישוב יחס המילים הייחודיות לפוסט
features_df = add_avg_word_length(posts_df, features_df)  # חישוב אורך המילים הממוצע בכל פוסט
features_df = add_tfidf_score(posts_df, features_df)  # חישוב ציון TF-IDF עבור כל פוסט


# *** קבוצה 3: סנטימנט ***
# מחשבים סנטימנט באמצעות מספר ספריות שונות ומשלבים את התוצאות.
features_df = add_sentiment_score(posts_df, features_df)  # חישוב סנטימנט בעזרת TextBlob
features_df = add_vader_sentiment_score(posts_df, features_df)  # חישוב סנטימנט בעזרת Vader
features_df = add_flair_sentiment_score(posts_df, features_df)  # חישוב סנטימנט בעזרת Flair
features_df = add_bert_sentiment_score(posts_df, features_df)  # חישוב סנטימנט בעזרת BERT
features_df = add_final_sentiment_score(features_df)  # חישוב סנטימנט סופי לפי "הרוב קובע"

# *** קבוצה 4: פורמליות ואיכות כתיבה ***
# מחשבים מדדי פורמליות, איכות כתיבה ויחסי סימני פיסוק.
features_df = add_formality_score(posts_df, features_df)  # חישוב ציון פורמליות בעזרת Flesch Reading Ease
features_df = add_alternative_formality_score(posts_df, features_df)  # חישוב מדד פורמליות אלטרנטיבי (Gunning Fog Index)
features_df = add_combined_formality_features(features_df)  # שילוב מדדי הפורמליות למדד משוקלל
features_df = add_punctuation_ratio(posts_df, features_df)  # חישוב יחס סימני פיסוק לכל פוסט
features_df = add_normalized_punctuation_features(posts_df, features_df)  # חישוב יחס סימני פיסוק מנורמלים לפי קטגוריות

#***איכות כתיבה***
features_df = add_writing_quality_score(posts_df, features_df)  # חישוב ציון איכות כתיבה (שגיאות כתיב ושגיאות דקדוק)


# *** קבוצה 5: זמנים (עבר/הווה/עתיד) ***
# מחשבים את חלוקת זמני הפעלים עבור כל פוסט.
features_df = add_verb_tense_distribution(posts_df, features_df)  # חישוב חלוקת זמני פעלים (עבר, הווה, עתיד) בעזרת NLTK
features_df = add_verb_tense_distribution_alternative(posts_df, features_df)  # חישוב חלוקת זמני פעלים (עבר, הווה, עתיד) בעזרת Stanza
features_df = add_combined_tense_distribution(features_df)  # ממוצע בין שתי הפונקציות של חלוקת זמני פעלים

# הדפסת זמן הסיום
print(f"Function 'example_function' finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# *** פונקציות שלא בשימוש ***
# פונקציות אלו אינן מופעלות בקוד אך ייתכן ונשתמש בהן בהמשך.
# features_df = add_most_common_word(posts_df, features_df)  # מחשבת את המילה הנפוצה ביותר בכל פוסט
# features_df = add_grammar_error_ratio(posts_df, features_df)  # יחס שגיאות דקדוק לכל פוסט


# ייצוא טבלת הפיצ'רים
features_df.to_csv("features_table100Ktest1.csv", index=False)
print("Features table exported successfully to 'features_table100Ktest1.csv'.")

# ייצוא טבלת הפוסטים
#posts_df.to_csv("posts_table100Ktest1.csv", index=False)
#print("Posts table exported successfully to 'posts_table100Ktest1.csv'.")







