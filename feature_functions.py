
from imports import *
from utils import *


#פונקציה טעינה והכנת נתונים
def load_and_prepare_data(base_repo_dir, file_name="train_10000.json"):
    """
    טוען נתונים מקובץ JSON, מכין את הנתונים לתהליך עיבוד.

    Args:
    - base_repo_dir (str): הנתיב לתיקיית הבסיס של הפרויקט.
    - file_name (str): שם קובץ הנתונים (ברירת מחדל: "train_1000.json").

    Returns:
    - pd.DataFrame: DataFrame מוכן לעיבוד.
    """
    # הגדרת הנתיב לקובץ
    file_path = os.path.join(base_repo_dir, "data", file_name)

    # טעינת נתונים ל-DataFrame
    train_df = json_to_df(json_file=file_path)
    print(f"Initial DataFrame shape: {train_df.shape}")
    print(train_df.head(2))  # הצגה של השורות הראשונות לבדיקה

    # הכנת טבלת פיצ'רים בסיסית
    basic_feature_df = train_df[["age", "gender", "post"]].copy()

    # הוספת עמודת אינדקס ייחודי לכל פוסט
    basic_feature_df["post_index"] = basic_feature_df.index
    print(f"Basic Feature DataFrame shape: {basic_feature_df.shape}")
    print(basic_feature_df.head())

    return basic_feature_df

# פונקציה פנימית לחישוב יחס מילות הקישור בטקסט
def stop_word_ratio(text):
    if not text:
        return 0
    words = text.split()
    stop_word_count = sum(1 for word in words if word.lower() in stop_words)
    return stop_word_count / len(words)

# הוספת הפיצ'ר לטבלה
def add_stop_word_ratio_feature(df):
    """
    מחשבת ומוסיפה את הפיצ'ר stop_word_ratio לטבלה נתונה.
    """
    df['stop_word_ratio'] = df['post'].apply(stop_word_ratio)
    print("DataFrame after adding stop_word_ratio feature:")
    print(df[['post', 'stop_word_ratio']].head())
    return df

def remove_stop_words_from_posts(df):
    """
    מנקה מילות קישור מעמודת 'post' ומעדכנת את הטבלה.

    Args:
    - df (pd.DataFrame): טבלת הנתונים המכילה עמודת 'post'.

    Returns:
    - pd.DataFrame: טבלה מעודכנת לאחר הסרת מילות הקישור.
    """
    # פונקציה פנימית להסרת מילות קישור מטקסט
    def remove_stop_words(text):
        if not text:  # אם הטקסט ריק
            return text
        words = text.split()
        # סינון המילים שהן מילות קישור
        cleaned_text = ' '.join([word for word in words if word.lower() not in stop_words])
        return cleaned_text

    # עדכון עמודת 'post' לאחר ניקוי מילות הקישור
    df['post'] = df['post'].apply(remove_stop_words)
    print("DataFrame after removing stop words:")
    print(df[['post']].head())

    return df

#פונקציה לבדיקת ערכים חסרים בעמדת הגיל
def check_missing_ages(df):
    """
    בודקת ערכים חסרים בעמודת 'age' ומדפיסה את התוצאה.

    Args:
    - df (pd.DataFrame): טבלת הנתונים המכילה את עמודת 'age'.

    Returns:
    - None
    """
    missing_ages = df['age'].isnull().sum()  # חישוב מספר הערכים החסרים
    if missing_ages > 0:
        print(f"There are {missing_ages} missing values in the 'age' column.")
    else:
        print("No missing values in the 'age' column.")

#פונקציה למציאת המילה הנפוצה ביותר
def add_most_common_word(df):
    """
    מוסיפה עמודת 'most_common_word' עם המילה הנפוצה ביותר בכל פוסט.

    Args:
    - df (pd.DataFrame): טבלת הנתונים המכילה עמודת 'post'.

    Returns:
    - pd.DataFrame: הטבלה המעודכנת עם עמודת most_common_word.
    """
    df['most_common_word'] = df['post'].apply(get_most_common_word_simple)
    print("Added most_common_word column:")
    print(df[['post', 'most_common_word']].head())
    return df

#פונקציה לחישוב מספר המילים
def add_word_count(df):
    """
    מוסיפה עמודת 'word_count' עם מספר המילים בכל פוסט.

    Args:
    - df (pd.DataFrame): טבלת הנתונים המכילה עמודת 'post'.

    Returns:
    - pd.DataFrame: הטבלה המעודכנת עם עמודת word_count.
    """
    df['word_count'] = df['post'].apply(lambda x: len(x.split()) if pd.notnull(x) else 0)
    print("Added word_count column:")
    print(df[['post', 'word_count']].head())
    return df

#פונקציה לחישוב יחס מילים ייחודיות בתוך הטקסט
def add_unique_word_ratio(df):
    """
    מחשבת ומוסיפה עמודת יחס מילים ייחודיות (unique_word_ratio) לטבלה.

    Args:
    - df (pd.DataFrame): טבלת הנתונים המכילה עמודת 'post'.

    Returns:
    - pd.DataFrame: הטבלה המעודכנת עם עמודת unique_word_ratio.
    """
    from nltk.tokenize import word_tokenize  # ייבוא פנימי לטובת הפונקציה

    # הוספת עמודת unique_word_ratio
    df['unique_word_ratio'] = df['post'].apply(
        lambda text: 0 if pd.isnull(text) or len(word_tokenize(text)) == 0
        else len(set(word_tokenize(text))) / len(word_tokenize(text))
    )

    # הדפסת תוצאה לדוגמה
    print("DataFrame after adding unique_word_ratio:")
    print(df[['post', 'unique_word_ratio']].head())

    return df

#פונקציה להוספת TF-IDF Score

def add_tfidf_score(df):
    """
    מחשבת ומוסיפה עמודת tfidf_score לטבלה, המבוססת על עמודת 'post'.

    Args:
    - df (pd.DataFrame): טבלת הנתונים המכילה עמודת 'post'.

    Returns:
    - pd.DataFrame: הטבלה המעודכנת עם עמודת tfidf_score.
    """
    # יצירת אובייקט TF-IDF
    tfidf_vectorizer = TfidfVectorizer()

    # חישוב מטריצת TF-IDF עבור עמודת 'post'
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['post'])

    # חישוב ממוצע הציונים של כל טקסט והוספת עמודת tfidf_score
    df['tfidf_score'] = tfidf_matrix.mean(axis=1).A1  # A1 ממיר את התוצאה ל-vector

    # הדפסת דוגמה מתוך התוצאה
    print("DataFrame after adding tfidf_score:")
    print(df[['post', 'tfidf_score']].head())

    return df

#פונקציה להוספת santiment score
def add_sentiment_score(df):
    """
    מחשבת ומוסיפה עמודת sentiment_score לטבלה, המבוססת על עמודת 'post'.

    Args:
    - df (pd.DataFrame): טבלת הנתונים המכילה עמודת 'post'.

    Returns:
    - pd.DataFrame: הטבלה המעודכנת עם עמודת sentiment_score.
    """
    # חישוב sentiment score עבור כל טקסט
    df['sentiment_score'] = df['post'].apply(
        lambda text: TextBlob(text).sentiment.polarity if pd.notnull(text) else 0
    )

    # הדפסת דוגמה מתוך התוצאה
    print("DataFrame after adding sentiment_score:")
    print(df[['post', 'sentiment_score']].head())

    return df

def add_vader_sentiment_label(df):
    """
    מחשבת ומוסיפה עמודת תווית סנטימנט (vader_sentiment_label) לטבלה, המבוססת על עמודת 'post'.

    Args:
    - df (pd.DataFrame): טבלת הנתונים המכילה עמודת 'post'.

    Returns:
    - pd.DataFrame: הטבלה המעודכנת עם עמודת vader_sentiment_label.
    """
    # יצירת אובייקט SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()

    # פונקציה פנימית לחישוב תווית סנטימנט
    def vader_sentiment_label(text):
        if not text:  # אם הטקסט ריק
            return "Neutral"

        # חישוב סנטימנט
        sentiment_scores = analyzer.polarity_scores(text)
        compound_score = sentiment_scores['compound']

        # קביעת תווית
        if compound_score > 0.05:
            return "Positive"
        elif compound_score <= -0.05:
            return "Negative"
        else:
            return "Neutral"

    # הוספת עמודת vader_sentiment_label לטבלה
    df['vader_sentiment_label'] = df['post'].apply(vader_sentiment_label)

    # הדפסת דוגמה מתוך התוצאה
    print("DataFrame after adding VADER sentiment label:")
    print(df[['post', 'vader_sentiment_label']].head())

    return df

#פונקציה להוספת Formality Score
def add_formality_score(df):
    """
    מחשבת ומוסיפה עמודת formality_score לטבלה, המבוססת על רמת הקריאות של הטקסט.

    Args:
    - df (pd.DataFrame): טבלת הנתונים המכילה עמודת 'post'.

    Returns:
    - pd.DataFrame: הטבלה המעודכנת עם עמודת formality_score.
    """
    def calculate_formality(text):
        """
        מחשבת את ה-formality score המבוסס על מדד הקריאות של הטקסט.
        """
        if not text:  # אם הטקסט ריק
            return 0.5  # ערך ניטרלי
        readability_score = textstat.flesch_reading_ease(text)  # חישוב קריאות
        # נורמליזציה לערכים בין 0 ל-1
        formality_score = max(0, min(1, 1 - (readability_score / 100)))
        return formality_score

    # חישוב formality score עבור כל טקסט
    df['formality_score'] = df['post'].apply(calculate_formality)

    # הדפסת דוגמה מתוך התוצאה
    print("DataFrame after adding formality_score:")
    print(df[['post', 'formality_score']].head())

    return df

#פונקצית יחס סימני הפיסוק
def add_punctuation_ratio(df):
    """
    מחשבת ומוסיפה עמודת punctuation_ratio לטבלה, המבוססת על יחס סימני הפיסוק למספר המילים.

    Args:
    - df (pd.DataFrame): טבלת הנתונים המכילה עמודת 'post'.

    Returns:
    - pd.DataFrame: הטבלה המעודכנת עם עמודת punctuation_ratio.
    """
    def punctuation_ratio(text):
        """
        מחשבת את יחס סימני הפיסוק בטקסט.
        """
        if not text:  # אם הטקסט ריק
            return 0
        punctuation_count = len(re.findall(r'[.,!?;:]', text))  # חישוב סימני פיסוק
        word_count = len(text.split())  # חישוב מספר המילים
        return punctuation_count / word_count if word_count > 0 else 0

    # הוספת עמודת punctuation_ratio
    df['punctuation_ratio'] = df['post'].apply(punctuation_ratio)

    # הדפסת דוגמה מתוך התוצאה
    print("DataFrame after adding punctuation_ratio:")
    print(df[['post', 'punctuation_ratio']].head())

    return df

#פונקציה להערכת דיוק הפיסוק
def add_punctuation_correctness_score(df):
    """
    מחשבת ומוסיפה עמודת punctuation_correctness_score לטבלה, המבוססת על בדיקת שגיאות פיסוק.

    Args:
    - df (pd.DataFrame): טבלת הנתונים המכילה עמודת 'post'.

    Returns:
    - pd.DataFrame: הטבלה המעודכנת עם עמודת punctuation_correctness_score.
    """


    def punctuation_correctness_score(text):
        """
        מחשבת את ניקוד הפיסוק הנכון עבור טקסט.
        """
        if not text:  # אם הטקסט ריק
            return 0

        matches = tool.check(text)  # בדיקת הטקסט
        # סופרים רק שגיאות הקשורות לפיסוק ורווחים
        total_punctuation_errors = sum(
            1 for match in matches if "PUNCTUATION" in match.ruleId or "WHITESPACE" in match.ruleId
        )

        # מחשבים את הניקוד (1 - יחס השגיאות למספר המילים)
        word_count = len(text.split())
        score = max(0, 1 - (total_punctuation_errors / word_count)) if word_count > 0 else 0
        return score

    # חישוב punctuation correctness score לכל טקסט
    df['punctuation_correctness_score'] = df['post'].apply(punctuation_correctness_score)

    # הדפסת דוגמה מתוך התוצאה
    print("DataFrame after adding punctuation_correctness_score:")
    print(df[['post', 'punctuation_correctness_score']].head())

    return df

#פונקצית ציון שגיאות דקדוק
def add_grammar_error_ratio(df):
    """
    מחשבת ומוסיפה עמודת grammar_error_ratio לטבלה, המבוססת על יחס שגיאות דקדוק לכל מילה בטקסט.

    Args:
    - df (pd.DataFrame): טבלת הנתונים המכילה עמודת 'post'.

    Returns:
    - pd.DataFrame: הטבלה המעודכנת עם עמודת grammar_error_ratio.
    """
    import language_tool_python
    tool = language_tool_python.LanguageTool('en-US')  # יצירת כלי בדיקת דקדוק

    def grammar_error_ratio(text):
        """
        מחשבת את יחס שגיאות הדקדוק בטקסט.
        """
        if not text:  # אם הטקסט ריק
            return 0
        matches = tool.check(text)  # בדיקת הטקסט
        total_words = len(text.split())  # חישוב מספר המילים בטקסט
        return len(matches) / total_words if total_words > 0 else 0  # יחס

    # הוספת עמודת grammar_error_ratio
    df = df.copy()  # יוצרים עותק כדי להימנע מ-Warning
    df['grammar_error_ratio'] = df['post'].apply(grammar_error_ratio)

    # הדפסת דוגמה מתוך התוצאה
    print("DataFrame after adding grammar_error_ratio:")
    print(df[['post', 'grammar_error_ratio']].head())

    return df

#פונקצית חישוב אורך מילה ממוצע
def add_avg_word_length(df):
    """
    מחשבת ומוסיפה עמודת avg_word_length לטבלה, המבוססת על אורך המילים הממוצע בטקסט.

    Args:
    - df (pd.DataFrame): טבלת הנתונים המכילה עמודת 'post'.

    Returns:
    - pd.DataFrame: הטבלה המעודכנת עם עמודת avg_word_length.
    """
    def average_word_length(text):
        """
        מחשבת את אורך המילים הממוצע בטקסט.
        """
        if not text:  # אם הטקסט ריק
            return 0.0
        words = text.split()  # פיצול הטקסט למילים
        # חישוב אורך כל המילים (מתעלמים מסימני פיסוק מסוימים בסוף המילה)
        total_length = sum(len(word.strip(".,!?;:")) for word in words)
        return total_length / len(words) if len(words) > 0 else 0.0  # ממוצע

    # יצירת עותק כדי להימנע מ-Warning
    df = df.copy()

    # הוספת עמודת avg_word_length
    df['avg_word_length'] = df['post'].apply(average_word_length)

    # הדפסת דוגמה מתוך התוצאה
    print("DataFrame after adding avg_word_length:")
    print(df[['post', 'avg_word_length']].head())

    return df

#פונקצית חישוב יחס סימני פיסוק מפורט
def add_normalized_punctuation_features(df):
    """
    מחשבת ומוסיפה עמודות פיצ'רים מנורמלים הקשורים לסימני פיסוק:
    - יחס סימני שאלה
    - יחס סימני קריאה
    - יחס פסיקים
    - יחס נקודות

    Args:
    - df (pd.DataFrame): טבלת הנתונים המכילה עמודת 'post'.

    Returns:
    - pd.DataFrame: הטבלה המעודכנת עם העמודות הבאות:
        'question_mark_ratio', 'exclamation_mark_ratio',
        'comma_ratio', 'period_ratio'.
    """
    def punctuation_feature(text):
        """
        מחשבת פיצ'רים מנורמלים הקשורים לסימני פיסוק עבור טקסט נתון.
        """
        if not text:  # אם הטקסט ריק
            return {'question_mark_ratio': 0,
                    'exclamation_mark_ratio': 0,
                    'comma_ratio': 0,
                    'period_ratio': 0}

        # סימני הפיסוק למעקב
        punctuation_marks = {'?': 'question_mark_ratio',
                             '!': 'exclamation_mark_ratio',
                             ',': 'comma_ratio',
                             '.': 'period_ratio'}

        word_count = len(text.split())  # מספר המילים בטקסט
        if word_count == 0:  # אם אין מילים, כל היחסים הם 0
            return {name: 0 for name in punctuation_marks.values()}

        # חישוב יחס סימני הפיסוק
        punctuation_ratios = {name: text.count(mark) / word_count for mark, name in punctuation_marks.items()}
        return punctuation_ratios

    # חישוב הפיצ'רים והוספתם לטבלה
    df = df.copy()  # יוצרים עותק כדי להימנע מ-Warning
    punctuation_features = df['post'].apply(punctuation_feature).apply(pd.Series)
    df = pd.concat([df, punctuation_features], axis=1)

    # הדפסת דוגמה מתוך התוצאה
    print("DataFrame after adding normalized punctuation features:")
    print(df[['post', 'question_mark_ratio', 'exclamation_mark_ratio',
              'comma_ratio', 'period_ratio']].head())

    return df

#פונקציה להוספת יחס זמני פעלים
def add_verb_tense_distribution(df):
    """
    מחשבת ומוסיפה עמודות הקשורות לחלוקת זמני הפעלים בטקסט:
    - past_ratio: יחס פעלים בזמן עבר
    - present_ratio: יחס פעלים בזמן הווה
    - future_ratio: יחס פעלים בזמן עתיד

    Args:
    - df (pd.DataFrame): טבלת הנתונים המכילה עמודת 'post'.

    Returns:
    - pd.DataFrame: הטבלה המעודכנת עם עמודות:
        'past_ratio', 'present_ratio', 'future_ratio'.
    """
    import spacy
    nlp = spacy.load("en_core_web_sm")  # מודל NLP של spaCy

    def verb_tense_distribution(text):
        """
        מחשבת את חלוקת זמני הפעלים עבור טקסט נתון.
        """
        if not text:  # אם הטקסט ריק
            return {"past_ratio": 0, "present_ratio": 0, "future_ratio": 0}

        # ניתוח הטקסט באמצעות spaCy
        doc = nlp(text)
        tense_counts = {"past": 0, "present": 0, "future": 0}

        # ספירת פעלים לפי זמן
        for token in doc:
            if token.tag_ in ["VBD", "VBN"]:  # זמן עבר
                tense_counts["past"] += 1
            elif token.tag_ in ["VBG", "VBP", "VBZ"]:  # זמן הווה
                tense_counts["present"] += 1
            elif token.tag_ == "MD" and token.text.lower() == "will":  # זמן עתיד
                tense_counts["future"] += 1

        # חישוב סך כל הפעלים
        total_verbs = sum(tense_counts.values())

        # חישוב יחס הזמנים
        if total_verbs > 0:
            tense_ratios = {
                "past_ratio": tense_counts["past"] / total_verbs,
                "present_ratio": tense_counts["present"] / total_verbs,
                "future_ratio": tense_counts["future"] / total_verbs,
            }
        else:
            tense_ratios = {"past_ratio": 0, "present_ratio": 0, "future_ratio": 0}

        return tense_ratios

    # חישוב פיצ'רים והוספתם לטבלה
    df = df.copy()  # עותק כדי להימנע מ-Warning
    tense_features = df['post'].apply(verb_tense_distribution).apply(pd.Series)
    df = pd.concat([df, tense_features], axis=1)

    # הדפסת דוגמה מתוך התוצאה
    print("DataFrame after adding verb tense distribution:")
    print(df[['post', 'past_ratio', 'present_ratio', 'future_ratio']].head())

    return df
