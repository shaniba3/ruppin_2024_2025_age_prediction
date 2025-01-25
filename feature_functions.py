
from imports import *
from utils import *

# פונקציה לעדכון ושמירת טבלה אחרי כל שלב
def export_table_to_parquet(dataframe, file_name):
    dataframe.to_parquet(file_name, index=False)
    print(f"Table exported successfully to '{file_name}'.")


# פונקציה טעינה והכנת נתונים
def load_and_prepare_data(base_repo_dir, file_name="split_files/train_split_1.json"):
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


def add_stop_word_ratio (posts_df, features_df):
    # הכנת רשימת Stop Words כ-Set לביצועים טובים יותר
    stop_words_set = set(stop_words)

    # חישוב יחס מילות קישור לכל טקסט
    def stop_word_ratio(post):
        if not post:
            return 0
        words = post.split()
        stop_word_count = sum(word.lower() in stop_words_set for word in words)
        return stop_word_count / len(words) if len(words) > 0 else 0

    # שימוש באפליקציה ווקטורית (Vectorized Application)
    features_df["stop_word_ratio"] = posts_df["post"].apply(stop_word_ratio)
    return features_df



# מסיר מילות קישור מהפוסטים ומעדכן את עמודת הפוסטים בטבלה
def remove_stop_words(posts_df):
    """
    מסירה מילות קישור (Stop Words) מכל הפוסטים בטבלת הנתונים.
    """
    import re

    # שימוש ב-Tokenizer במקום split
    stop_words_pattern = re.compile(r'\b(?:' + '|'.join(stop_words) + r')\b', flags=re.IGNORECASE)

    # מסירים את מילות הקישור בצורה וקטורית לכל הפוסטים
    posts_df['post'] = posts_df['post'].fillna('')  # טיפול בטקסטים ריקים
    posts_df['post'] = posts_df['post'].apply(lambda x: stop_words_pattern.sub('', x).strip())

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
    features_df["word_count"] = posts_df["post"].str.split().str.len().fillna(0)
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
def add_vader_sentiment_score(posts_df, features_df):
    """
    מחשבת את ציון ה-Compound של סנטימנט (VADER) עבור כל פוסט ומוסיפה לטבלת הפיצ'רים.
    """
    analyzer = SentimentIntensityAnalyzer()

    def vader_sentiment_score(text):
        if not text:
            return 0  # ערך ניטרלי במקרה של טקסט ריק
        sentiment_scores = analyzer.polarity_scores(text)
        return sentiment_scores['compound']  # מחזיר את הציון המספרי

    features_df["vader_sentiment_score"] = posts_df["post"].apply(vader_sentiment_score)

    print("Added vader_sentiment_score column:")
    print(features_df[["post_index", "vader_sentiment_score"]].head())

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
        if not text or pd.isnull(text):
            return 0  # ערך ניטרלי במקרה של טקסט ריק או חסר

        # חלוקת הטקסט למקטעים בגודל 512 תווים
        segments = [text[i:i+512] for i in range(0, len(text), 512)]
        sentiment_scores = []

        for segment in segments:
            result = sentiment_pipeline(segment)
            sentiment_score = result[0]["score"]  # ציון הסנטימנט
            sentiment_value = result[0]["label"]  # חיובי/שלילי
            # חישוב הציון הסופי
            sentiment_scores.append(sentiment_score if sentiment_value == "POSITIVE" else -sentiment_score)

        # החזרת ממוצע הציונים במקרה של טקסט ארוך
        return sum(sentiment_scores) / len(sentiment_scores)

    # הוספת הפיצ'ר לטבלת הפיצ'רים
    features_df["bert_sentiment_score"] = posts_df["post"].apply(calculate_bert_sentiment)

    print("Added sentiment score (BERT):")
    print(features_df[["post_index", "bert_sentiment_score"]].head())

    return features_df


def add_final_sentiment_score(features_df):
    """
    מחשבת את הציון הסופי על פי שיטת "הרוב קובע".
    אם יש תיקו:
    - תיקו בין שלילי ונייטרלי מחזיר -1.
    - תיקו בין חיובי ונייטרלי מחזיר 1.
    - תיקו בין חיובי לשלילי מחזיר 0.
    """
    # הפיכת ציוני הסנטימנט לכלי החלטה (1 חיובי, 0 ניטרלי, -1 שלילי)
    def convert_to_decision(score):
        if score > 0.05:
            return 1  # חיובי
        elif score < -0.05:
            return -1  # שלילי
        else:
            return 0  # ניטרלי

    # הגדרת עמודות הסנטימנט הקיימות
    sentiment_columns = [
        "sentiment_score",         # TextBlob
        "vader_sentiment_score",   # VADER
        "flair_sentiment_score",   # Flair
        "bert_sentiment_score"     # BERT
    ]

    # החלת ההחלטה על כל ציוני הסנטימנט
    for col in sentiment_columns:
        features_df[f"{col}_decision"] = features_df[col].apply(convert_to_decision)

    # פונקציה שמחשבת את הרוב
    def majority_vote(row):
        # ספירה של כל קטגוריה
        positive = sum(1 for val in row if val == 1)
        neutral = sum(1 for val in row if val == 0)
        negative = sum(1 for val in row if val == -1)

        # קביעת התוצאה לפי רוב
        if positive > negative and positive > neutral:
            return 1
        elif negative > positive and negative > neutral:
            return -1
        elif neutral > positive and neutral > negative:
            return 0

        # טיפולי תיקו:
        if positive == negative:  # חיובי ושלילי שווים
            return 0
        elif negative == neutral:  # שלילי ונייטרלי שווים
            return -1
        elif positive == neutral:  # חיובי ונייטרלי שווים
            return 1

    # החלת ההחלטה על כל השורות
    features_df["final_sentiment"] = features_df[
        [f"{col}_decision" for col in sentiment_columns]
    ].apply(majority_vote, axis=1)

    print("Added final_sentiment column:")
    print(features_df[["post_index", "final_sentiment"]].head())

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
    מחשבת מדד פורמליות אלטרנטיבי (Gunning Fog Index) וממירה אותו לטווח 0-1.
    """
    def calculate_gunning_fog(text):
        if not text:  # אם אין טקסט, ערך ברירת מחדל
            return 0.5
        raw_score = textstat.gunning_fog(text)  # מחשב את הציון המקורי
        normalized_score = max(0, min(1, 1 - (raw_score / 20)))  # מנרמל לטווח 0-1
        return normalized_score

    features_df["formality_score_gunning_fog"] = posts_df["post"].apply(calculate_gunning_fog)

    print("Added alternative formality score (Gunning Fog):")
    print(features_df[["post_index", "formality_score_gunning_fog"]].head())

    return features_df


def add_combined_formality_features( features_df):

    def calculate_combined_formality(f_score, g_score):
        return (f_score + g_score) / 2

    def determine_formality_label(f_score, g_score):
        avg_score = (f_score + g_score) / 2
        if avg_score > 0.5:
            return 1  # פורמלי
        elif avg_score < 0.2:
            return -1  # לא פורמלי
        else:
            return 0  # ביניים

    # חישוב העמודה הרציפה
    features_df["combined_formality_score"] = features_df.apply(
        lambda row: calculate_combined_formality(
            row["formality_score"], row["formality_score_gunning_fog"]
        ),
        axis=1,
    )

    # חישוב העמודה הקטגוריאלית
    features_df["combined_formality_label"] = features_df.apply(
        lambda row: determine_formality_label(
            row["formality_score"], row["formality_score_gunning_fog"]
        ),
        axis=1,
    )

    # הדפסת נתוני העמודות החדשות
    print("Added combined_formality_score and combined_formality_label columns:")
    print(features_df[["post_index", "combined_formality_score", "combined_formality_label"]].head())

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




#חישוב ציון כתיב (גם דקדוק וגם כתיב)
def calculate_writing_quality(text):

    if not text:  # אם אין טקסט, ציון ברירת מחדל
        return 0.5


    matches = tool.check(text)

    # חישוב מספר השגיאות לפי קטגוריה
    spell_errors = sum(1 for match in matches if 'SPELLING' in match.ruleId)
    grammar_errors = sum(1 for match in matches if 'GRAMMAR' in match.ruleId or 'PUNCTUATION' in match.ruleId)

    # חישוב סך כל השגיאות
    total_errors = spell_errors + grammar_errors
    word_count = len(text.split())

    # חישוב ציון איכות הכתיבה
    if word_count == 0:  # למקרה של טקסט ריק
        return 0.5
    writing_quality_score = max(0, 1 - (total_errors / word_count))

    return writing_quality_score


def add_writing_quality_score(posts_df, features_df):

    features_df["writing_quality_score"] = posts_df["post"].apply(calculate_writing_quality)
    print("Added writing_quality_score column:")
    print(features_df[["post_index", "writing_quality_score"]].head())
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

def add_combined_tense_distribution(features_df):
    """
    מחשבת ממוצע בין שתי הפונקציות של חלוקת זמני הפעלים (spaCy ו־Stanza).
    """
    # בודקת אם כל העמודות הדרושות קיימות
    required_columns = [
        'past_ratio', 'present_ratio', 'future_ratio',
        'past_ratio_stanza', 'present_ratio_stanza', 'future_ratio_stanza'
    ]
    for col in required_columns:
        if col not in features_df.columns:
            raise ValueError(f"Column '{col}' is missing from features_df. Make sure both tense functions ran successfully.")

    # חישוב ממוצעים עבור כל זמן
    features_df['past_ratio_combined'] = (
        features_df['past_ratio'] + features_df['past_ratio_stanza']
    ) / 2
    features_df['present_ratio_combined'] = (
        features_df['present_ratio'] + features_df['present_ratio_stanza']
    ) / 2
    features_df['future_ratio_combined'] = (
        features_df['future_ratio'] + features_df['future_ratio_stanza']
    ) / 2

    # הדפסת דוגמה של התוצאות
    print("DataFrame after adding combined tense distribution:")
    print(features_df[['post_index', 'past_ratio_combined', 'present_ratio_combined', 'future_ratio_combined']].head())

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


