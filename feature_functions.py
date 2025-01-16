
from imports import *
from utils import *



# פונקציה טעינה והכנת נתונים
def load_and_prepare_data(base_repo_dir, file_name="train_10000.json"):
    # הגדרת הנתיב לקובץ
    file_path = os.path.join(base_repo_dir, "data", file_name)

    # טעינת נתונים ל-DataFrame
    train_df = json_to_df(json_file=file_path)
    print(f"Initial DataFrame shape: {train_df.shape}")
    print(train_df.head(2))  # הצגה של השורות הראשונות לבדיקה

    # הכנת טבלת הפוסטים המקורית
    posts_df = train_df[["age", "post"]].copy()
    posts_df["post_index"] = posts_df.index
    print(f"Posts DataFrame shape: {posts_df.shape}")
    print(posts_df.head())

    # יצירת טבלת פיצ'רים ריקה עם אינדקס וגיל בלבד
    features_df = posts_df[["post_index", "age"]].copy()
    print(f"Features DataFrame shape: {features_df.shape}")
    print(features_df.head())

    return posts_df, features_df


def add_stop_word_ratio(posts_df, features_df):
    def stop_word_ratio(text):
        if not text:
            return 0
        words = text.split()
        stop_word_count = sum(1 for word in words if word.lower() in stop_words)
        return stop_word_count / len(words) if len(words) > 0 else 0

    # חישוב הפיצ'ר והתאמתו לטבלת הפיצ'רים
    features_df["stop_word_ratio"] = posts_df["post"].apply(stop_word_ratio)
    print("Features DataFrame after adding stop_word_ratio feature:")
    print(features_df[["post_index", "stop_word_ratio"]].head())

    return features_df


# מסיר מילות קישור מהפוסטים ומעדכן את עמודת הפוסטים בטבלה
def remove_stop_words(posts_df):
    posts_df = posts_df.copy()

    def clean_post(post):
        if not post:
            return post
        words = post.split()
        cleaned_text = ' '.join([word for word in words if word.lower() not in stop_words])
        return cleaned_text

    posts_df['post'] = posts_df['post'].apply(clean_post)
    print("Posts after removing stop words:")
    print(posts_df[['post']].head())

    return posts_df

# בודק ומדפיס אם יש ערכים חסרים בעמודת הגיל
def check_missing_ages(features_df):
    missing_ages = features_df['age'].isnull().sum()
    if missing_ages > 0:
        print(f"There are {missing_ages} missing values in the 'age' column.")
    else:
        print("No missing values in the 'age' column.")

# מחשב את המילה הנפוצה ביותר בכל פוסט ומעדכן בטבלת הפיצ'רים
def add_most_common_word(posts_df, features_df):
    """
    מחשבת את המילה הנפוצה ביותר בכל פוסט ומוסיפה את התוצאה לטבלת הפיצ'רים.
    """
    def get_most_common_word(text):
        # בדיקה אם הטקסט ריק או לא תקין
        if not text or len(text.strip()) == 0:
            return None
        words = text.split()
        return max(set(words), key=words.count)

    # הוספת עמודת most_common_word לטבלת הפיצ'רים
    features_df['most_common_word'] = posts_df['post'].apply(get_most_common_word)

    print("DataFrame after adding most_common_word feature:")
    print(features_df[['post_index', 'most_common_word']].head())
    return features_df


# מחשב את מספר המילים בכל פוסט ומעדכן בטבלת הפיצ'רים
def add_word_count(posts_df, features_df):
    features_df["word_count"] = posts_df["post"].apply(
        lambda text: len(text.split()) if pd.notnull(text) else 0
    )
    print("Added word_count column:")
    print(features_df[["post_index", "word_count"]].head())
    return features_df

# מחשב את יחס המילים הייחודיות ביחס לכל המילים בפוסט ומעדכן בטבלת הפיצ'רים
def add_unique_word_ratio(posts_df, features_df):
    from nltk.tokenize import word_tokenize

    def unique_word_ratio(text):
        if pd.isnull(text) or len(word_tokenize(text)) == 0:
            return 0
        tokens = word_tokenize(text)
        return len(set(tokens)) / len(tokens)

    features_df["unique_word_ratio"] = posts_df["post"].apply(unique_word_ratio)
    print("Added unique_word_ratio column:")
    print(features_df[["post_index", "unique_word_ratio"]].head())
    return features_df

# מחשב את הציון TF-IDF לכל פוסט ומעדכן בטבלת הפיצ'רים
def add_tfidf_score(posts_df, features_df):
    tfidf_vectorizer = TfidfVectorizer()

    # חישוב מטריצת TF-IDF עבור עמודת 'post'
    tfidf_matrix = tfidf_vectorizer.fit_transform(posts_df["post"])

    # הוספת עמודת tfidf_score לטבלת הפיצ'רים
    features_df["tfidf_score"] = tfidf_matrix.mean(axis=1).A1  # A1 ממיר את התוצאה ל-vector

    print("Added tfidf_score column:")
    print(features_df[["post_index", "tfidf_score"]].head())
    return features_df

# מחשב את ציון הסנטימנט (sentiment score) עבור כל פוסט ומעדכן בטבלת הפיצ'רים
def add_sentiment_score(posts_df, features_df):
    features_df["sentiment_score"] = posts_df["post"].apply(
        lambda text: TextBlob(text).sentiment.polarity if pd.notnull(text) else 0
    )
    print("Added sentiment_score column:")
    print(features_df[["post_index", "sentiment_score"]].head())
    return features_df

# מחשב את תווית הסנטימנט (sentiment label) באמצעות VADER ומעדכן בטבלת הפיצ'רים
def add_vader_sentiment_label(posts_df, features_df):
    analyzer = SentimentIntensityAnalyzer()

    def vader_sentiment_label(text):
        if not text:
            return "Neutral"
        sentiment_scores = analyzer.polarity_scores(text)
        compound_score = sentiment_scores['compound']
        if compound_score > 0.05:
            return "Positive"
        elif compound_score <= -0.05:
            return "Negative"
        else:
            return "Neutral"

    features_df["vader_sentiment_label"] = posts_df["post"].apply(vader_sentiment_label)
    print("Added vader_sentiment_label column:")
    print(features_df[["post_index", "vader_sentiment_label"]].head())
    return features_df

def add_flair_sentiment_score(posts_df, features_df):
    """
    מחשבת סנטימנט עם Flair ומוסיפה לטבלת הפיצ'רים עמודה ייעודית.
    """


    # יצירת המודל של Flair
    classifier = TextClassifier.load("sentiment")

    def calculate_flair_sentiment(text):
        if not text:
            return 0  # ערך ניטרלי במקרה של טקסט ריק
        sentence = Sentence(text)
        classifier.predict(sentence)
        sentiment_score = sentence.labels[0].score  # ציון הסנטימנט
        sentiment_value = sentence.labels[0].value  # חיובי/שלילי
        return sentiment_score if sentiment_value == "POSITIVE" else -sentiment_score

    # הוספת הפיצ'ר לטבלת הפיצ'רים
    features_df["flair_sentiment_score"] = posts_df["post"].apply(calculate_flair_sentiment)

    print("Added sentiment score (Flair):")
    print(features_df[["post_index", "flair_sentiment_score"]].head())

    return features_df

def add_bert_sentiment_score(posts_df, features_df):
    """
    מחשבת סנטימנט עם BERT ומוסיפה לטבלת הפיצ'רים עמודה ייעודית.
    """

    # יצירת pipeline לסיווג סנטימנט
    sentiment_pipeline = pipeline("sentiment-analysis")

    def calculate_bert_sentiment(text):
        """
        מחשבת את הסנטימנט עבור טקסט מסוים תוך טיפול במגבלות BERT.
        """
        if not text or pd.isnull(text):
            return 0  # ערך ניטרלי במקרה של טקסט ריק או חסר

        # חיתוך ל-512 תווים כדי להתאים למגבלת המודל
        trimmed_text = text[:512]

        # חישוב הסנטימנט
        result = sentiment_pipeline(trimmed_text)
        sentiment_score = result[0]["score"]  # ציון הסנטימנט
        sentiment_value = result[0]["label"]  # חיובי/שלילי

        # החזרת הציון בהתאמה לתווית
        return sentiment_score if sentiment_value == "POSITIVE" else -sentiment_score

    # הוספת הפיצ'ר לטבלת הפיצ'רים
    features_df["bert_sentiment_score"] = posts_df["post"].apply(calculate_bert_sentiment)

    # הדפסת עמודה לדוגמה לצורך בדיקה
    print("Added sentiment score (BERT):")
    print(features_df[["post_index", "bert_sentiment_score"]].head())

    return features_df




# מחשב את ציון ה-Formality Score ומעדכן בטבלת הפיצ'רים
def add_formality_score(posts_df, features_df):
    def calculate_formality(text):
        if not text:  # טקסט ריק
            return 0.5  # ערך ניטרלי
        readability_score = textstat.flesch_reading_ease(text)  # חישוב קריאות
        formality_score = max(0, min(1, 1 - (readability_score / 100)))  # נורמליזציה לערכים בין 0 ל-1
        return formality_score

    features_df["formality_score"] = posts_df["post"].apply(calculate_formality)
    print("Added formality_score column:")
    print(features_df[["post_index", "formality_score"]].head())
    return features_df

def add_alternative_formality_score(posts_df, features_df):
    """
    מחשבת מדד פורמליות נוסף (Gunning Fog Index) ומוסיפה לטבלת הפיצ'רים עם שם עמודה ייחודי.
    """
    def calculate_gunning_fog(text):
        if not text:
            return 0.5  # ערך ניטרלי במקרה של טקסט ריק
        return textstat.gunning_fog(text)  # חישוב Gunning Fog Index

    # הוספת הפיצ'ר לטבלת הפיצ'רים
    features_df["formality_score_gunning_fog"] = posts_df["post"].apply(calculate_gunning_fog)

    print("Added alternative formality score (Gunning Fog):")
    print(features_df[["post_index", "formality_score_gunning_fog"]].head())

    return features_df

# מחשב את יחס סימני הפיסוק ומעדכן בטבלת הפיצ'רים
def add_punctuation_ratio(posts_df, features_df):
    def punctuation_ratio(text):
        if not text:  # טקסט ריק
            return 0
        punctuation_count = len(re.findall(r'[.,!?;:]', text))  # מספר סימני הפיסוק
        word_count = len(text.split())  # מספר המילים
        return punctuation_count / word_count if word_count > 0 else 0  # יחס סימני פיסוק

    features_df["punctuation_ratio"] = posts_df["post"].apply(punctuation_ratio)
    print("Added punctuation_ratio column:")
    print(features_df[["post_index", "punctuation_ratio"]].head())
    return features_df

# מחשב את ניקוד דיוק הפיסוק ומעדכן בטבלת הפיצ'רים
def add_punctuation_correctness_score(posts_df, features_df):
    def punctuation_correctness_score(text):
        if not text:  # אם הטקסט ריק
            return 0
        matches = tool.check(text)  # בדיקת הטקסט
        total_punctuation_errors = sum(
            1 for match in matches if "PUNCTUATION" in match.ruleId or "WHITESPACE" in match.ruleId
        )
        word_count = len(text.split())  # חישוב מספר המילים
        return max(0, 1 - (total_punctuation_errors / word_count)) if word_count > 0 else 0

    features_df["punctuation_correctness_score"] = posts_df["post"].apply(punctuation_correctness_score)
    print("Added punctuation_correctness_score column:")
    print(features_df[["post_index", "punctuation_correctness_score"]].head())
    return features_df

# מחשב את יחס שגיאות הדקדוק ומעדכן בטבלת הפיצ'רים
def add_grammar_error_ratio(posts_df, features_df):
    import language_tool_python
    tool = language_tool_python.LanguageTool('en-US')

    def grammar_error_ratio(text):
        if not text:  # אם הטקסט ריק
            return 0
        matches = tool.check(text)  # בדיקת הטקסט
        total_words = len(text.split())  # חישוב מספר המילים
        return len(matches) / total_words if total_words > 0 else 0  # יחס שגיאות דקדוק

    features_df["grammar_error_ratio"] = posts_df["post"].apply(grammar_error_ratio)
    print("Added grammar_error_ratio column:")
    print(features_df[["post_index", "grammar_error_ratio"]].head())
    return features_df

# מחשב את אורך המילים הממוצע בטקסט ומעדכן בטבלת הפיצ'רים
def add_avg_word_length(posts_df, features_df):
    def average_word_length(text):
        if not text:  # אם הטקסט ריק
            return 0.0
        words = text.split()  # פיצול הטקסט למילים
        # חישוב אורך כל המילים (מתעלמים מסימני פיסוק בסוף המילה)
        total_length = sum(len(word.strip(".,!?;:")) for word in words)
        return total_length / len(words) if len(words) > 0 else 0.0  # ממוצע

    features_df["avg_word_length"] = posts_df["post"].apply(average_word_length)
    print("Added avg_word_length column:")
    print(features_df[["post_index", "avg_word_length"]].head())
    return features_df


# מחשב פיצ'רים מנורמלים הקשורים לסימני פיסוק ומעדכן בטבלת הפיצ'רים
def add_normalized_punctuation_features(posts_df, features_df):
    def punctuation_feature(text):
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

    # הוספת הפיצ'רים לטבלת הפיצ'רים
    punctuation_features = posts_df['post'].apply(punctuation_feature).apply(pd.Series)
    for column in punctuation_features.columns:
        features_df[column] = punctuation_features[column]

    print("Added normalized punctuation features:")
    print(features_df[
              ["post_index", "question_mark_ratio", "exclamation_mark_ratio", "comma_ratio", "period_ratio"]].head())
    return features_df

# מחשב פיצ'רים לחלוקת זמני הפעלים ומעדכן בטבלת הפיצ'רים
def add_verb_tense_distribution(posts_df, features_df):
    import spacy
    nlp = spacy.load("en_core_web_sm")  # מודל NLP של spaCy

    def verb_tense_distribution(text):
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

    # חישוב פיצ'רים לחלוקת זמני הפעלים והוספתם לטבלת הפיצ'רים
    tense_features = posts_df['post'].apply(verb_tense_distribution).apply(pd.Series)
    for column in tense_features.columns:
        features_df[column] = tense_features[column]

    print("Added verb tense distribution features:")
    print(features_df[["post_index", "past_ratio", "present_ratio", "future_ratio"]].head())
    return features_df

def add_verb_tense_distribution_alternative(posts_df, features_df):
    """
    מחשבת את חלוקת זמני הפעלים (עבר, הווה, עתיד) עבור כל פוסט ומוסיפה לטבלת הפיצ'רים.
    """
    def verb_tense_distribution_stanza(text):
        if not text:
            return {"past_ratio_stanza": 0, "present_ratio_stanza": 0, "future_ratio_stanza": 0}

        doc = nlp(text)
        tense_counts = {"past": 0, "present": 0, "future": 0}

        # עובר על כל המילים בטקסט ומחשב את תדירות הזמנים
        for sentence in doc.sentences:
            for word in sentence.words:
                if word.xpos in ["VBD", "VBN"]:  # זמן עבר
                    tense_counts["past"] += 1
                elif word.xpos in ["VBG", "VBP", "VBZ"]:  # זמן הווה
                    tense_counts["present"] += 1
                elif word.xpos == "MD" and word.text.lower() == "will":  # זמן עתיד
                    tense_counts["future"] += 1

        total_verbs = sum(tense_counts.values())
        if total_verbs > 0:
            tense_ratios = {
                "past_ratio_stanza": tense_counts["past"] / total_verbs,
                "present_ratio_stanza": tense_counts["present"] / total_verbs,
                "future_ratio_stanza": tense_counts["future"] / total_verbs,
            }
        else:
            tense_ratios = {"past_ratio_stanza": 0, "present_ratio_stanza": 0, "future_ratio_stanza": 0}

        return tense_ratios

    # מחשב את הזמנים ומוסיף לטבלת הפיצ'רים
    tense_features = posts_df['post'].apply(verb_tense_distribution_stanza).apply(pd.Series)
    features_df = pd.concat([features_df, tense_features], axis=1)

    print("DataFrame after adding alternative verb tense distribution:")
    print(features_df[['post_index', 'past_ratio_stanza', 'present_ratio_stanza', 'future_ratio_stanza']].head())

    return features_df

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
