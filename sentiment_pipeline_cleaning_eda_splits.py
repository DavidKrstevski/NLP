#%% md
# 
# # Sentiment Analysis
# 
#%% md
# ## 1.1) Load & Combine Data
#%%

# Load Data
import pandas as pd

df_yelp = pd.read_csv('./sentiment labelled sentences/yelp_labelled.txt', names=['sentence', 'label'], sep='\t')
df_amazon = pd.read_csv('./sentiment labelled sentences/amazon_cells_labelled.txt', names=['sentence', 'label'], sep='\t')
df_imdb = pd.read_csv('./sentiment labelled sentences/imdb_labelled.txt', names=['sentence', 'label'], sep='\t')

print("Yelp: ", df_yelp.shape)
print("Amazon: ", df_amazon.shape)
print("IMDB: ", df_imdb.shape)

df_all = pd.concat([df_yelp, df_amazon, df_imdb], ignore_index=True)
print("All: ", df_all.shape)
df_all.head()

#%% md
# ## 1.2) Explore Data
#%%

# Explore Data
import re, numpy as np, pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Dataset-wise label distribution (%)
for name, d in [('Yelp', df_yelp), ('Amazon', df_amazon), ('IMDB', df_imdb)]:
    percent = d['label'].value_counts(normalize=True, dropna=False).sort_index() * 100
    print(f"{name} Verteilung (%):\n", percent.round(2), "\n")

df = df_all.copy()

# Tokenizer (keeps simple English words and contractions like don't)
_token_re = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
def tok(s): 
    if not isinstance(s, str): return []
    return _token_re.findall(s.lower())

# Grundüberblick
print("Shape:", df.shape)
print("Missing:", df.isna().sum().to_dict())
print("\nLabel-Verteilung (Count/%)")
counts = df['label'].value_counts().sort_index()
pct = (df['label'].value_counts(normalize=True).sort_index()*100).round(2)
display(pd.DataFrame({'count': counts, 'percent': pct}))

# Kern-Metriken je Text
def stats(text):
    s = text if isinstance(text, str) else ""
    t = tok(s)
    n = len(t)
    return pd.Series({
        'char_len': len(s),
        'word_len': n,
        'avg_word_len': (sum(len(x) for x in t)/n) if n else 0.0,
        'stop_ratio': (sum(x in ENGLISH_STOP_WORDS for x in t)/(n or 1)),
        'ttr': (len(set(t))/(n or 1)),
    })

fe = df['sentence'].apply(stats)
df_e = pd.concat([df[['label']], fe], axis=1)

print("\nGesamt-Deskriptoren:")
display(df_e.describe().round(3).T)

print("\nMittelwerte je Label:")
display(df_e.groupby('label').mean(numeric_only=True).round(3))

# N-Grams (Top 10)
def top_ngrams(texts, n=1, k=10, drop_stop=True):
    c = Counter()
    for s in texts:
        t = tok(s)
        if drop_stop: t = [w for w in t if w not in ENGLISH_STOP_WORDS]
        grams = t if n==1 else [" ".join(t[i:i+n]) for i in range(len(t)-n+1)]
        c.update(grams)
    return pd.DataFrame(c.most_common(k), columns=[f'{n}-gram','count'])

print("\nTop Unigrams (gesamt):")
display(top_ngrams(df['sentence'], n=1, k=10))

print("\nTop Bigrams (gesamt):")
display(top_ngrams(df['sentence'], n=2, k=10))

for lbl in sorted(df['label'].unique()):
    subset = df.loc[df['label']==lbl, 'sentence']
    print(f"\nTop Unigrams (Label={lbl}):")
    display(top_ngrams(subset, n=1, k=10))
    print(f"\nTop Bigrams (Label={lbl}):")
    display(top_ngrams(subset, n=2, k=10))

# Vokabulargröße (ohne Stopwords)
def vocab_size(texts):
    v=set()
    for s in texts: v.update([w for w in tok(s) if w not in ENGLISH_STOP_WORDS])
    return len(v)

print("\nVokabular gesamt (ohne Stopwords):", vocab_size(df['sentence']))
print("Vokabular je Label (ohne Stopwords):")
display(df.groupby('label')['sentence'].apply(vocab_size).to_frame('vocab_size'))

#%% md
# ## 2.1) Text Cleaning
#%%

import re
import html
from html import unescape
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Optional: NLTK for lemmatization with safe fallback to stemming
use_lemmatize = True
try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/omw-1.4')
    _lemmatizer = WordNetLemmatizer()
except Exception:
    use_lemmatize = False

from nltk.stem import PorterStemmer
_stemmer = PorterStemmer()

# Basic contraction map (extend as needed)
CONTRACTIONS = {
    "ain't":"am not", "aren't":"are not", "can't":"cannot", "can't've":"cannot have",
    "could've":"could have", "couldn't":"could not", "didn't":"did not", "doesn't":"does not",
    "don't":"do not", "hadn't":"had not", "hasn't":"has not", "haven't":"have not",
    "he's":"he is", "she's":"she is", "it's":"it is", "i'm":"i am", "i've":"i have",
    "i'd":"i would", "i'll":"i will", "isn't":"is not", "let's":"let us", "mightn't":"might not",
    "mustn't":"must not", "shan't":"shall not", "shouldn't":"should not",
    "that's":"that is", "there's":"there is", "they're":"they are", "they've":"they have",
    "we're":"we are", "we've":"we have", "we'll":"we will", "weren't":"were not",
    "what's":"what is", "who's":"who is", "won't":"will not", "wouldn't":"would not",
    "you'd":"you would", "you'll":"you will", "you're":"you are", "y'all":"you all"
}

_url_re = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_html_tag_re = re.compile(r"<[^>]+>")
_num_re = re.compile(r"\b\d+(?:[.,]\d+)?\b")
_punct_re = re.compile(r"[\.,!?:;\-\(\)\[\]\{\}\"\'`~\/_\\\|@#\$%\^&\*\+=><]+" )
_token_re = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

def expand_contractions(text: str) -> str:
    def repl(match):
        c = match.group(0)
        return CONTRACTIONS.get(c.lower(), c)
    return re.sub(r"\b(" + "|".join(map(re.escape, CONTRACTIONS.keys())) + r")\b", repl, text, flags=re.IGNORECASE)

def basic_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    s = html.unescape(text)
    s = _url_re.sub(" ", s)                 # remove URLs
    s = _html_tag_re.sub(" ", s)            # remove HTML tags
    s = expand_contractions(s)              # expand contractions
    s = _num_re.sub(" ", s)                 # remove numbers
    s = _punct_re.sub(" ", s)               # remove punctuation
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize(text: str):
    return _token_re.findall(text)

def remove_stopwords(tokens, stopwords=ENGLISH_STOP_WORDS):
    return [t for t in tokens if t not in stopwords]

def normalize_tokens(tokens):
    out = []
    for t in tokens:
        if use_lemmatize:
            try:
                out.append(_lemmatizer.lemmatize(t))
            except Exception:
                out.append(_stemmer.stem(t))
        else:
            out.append(_stemmer.stem(t))
    return out

def clean_and_tokenize(text: str):
    s = basic_clean(text)
    toks = tokenize(s)
    toks = remove_stopwords(toks)
    toks = normalize_tokens(toks)
    return s, toks

# Apply cleaning
df['clean_text'], df['tokens'] = zip(*df['sentence'].map(clean_and_tokenize))

# Sentence lengths after cleaning (in tokens)
df['len_tokens'] = df['tokens'].apply(len)
df[['sentence','clean_text','label','len_tokens']].head()

#%% md
# ## 2.2) Exploratory Data Analysis (EDA)
#%%

import matplotlib.pyplot as plt
from collections import Counter

# Label distribution (bar chart)
label_counts = df['label'].value_counts().sort_index()
plt.figure()
label_counts.plot(kind='bar')
plt.title('Label Distribution')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()

# Word frequency plots (top 20 global, and per class top 20)
def plot_top_words(token_series, title, top_k=20):
    freq = Counter([t for tokens in token_series for t in tokens])
    top = freq.most_common(top_k)
    words, counts = zip(*top) if top else ([], [])
    plt.figure()
    plt.bar(range(len(words)), counts)
    plt.xticks(range(len(words)), words, rotation=90)
    plt.title(title)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

plot_top_words(df['tokens'], 'Top Words (All)', top_k=20)

for lbl in sorted(df['label'].unique()):
    plot_top_words(df.loc[df['label']==lbl, 'tokens'], f'Top Words (Label={lbl})', top_k=20)

# Word cloud for each class
try:
    from wordcloud import WordCloud
    for lbl in sorted(df['label'].unique()):
        text_blob = " ".join([" ".join(toks) for toks in df.loc[df['label']==lbl, 'tokens']])
        if text_blob.strip():
            wc = WordCloud(width=800, height=400, background_color='white').generate(text_blob)
            plt.figure(figsize=(10,5))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Word Cloud (Label={lbl})')
            plt.show()
        else:
            print(f"No tokens for label {lbl} to build word cloud.")
except Exception as e:
    print("WordCloud not available:", e)

# Sentence length distribution (histogram over cleaned token lengths)
plt.figure()
df['len_tokens'].plot(kind='hist', bins=30)
plt.title('Sentence Length Distribution (Cleaned Tokens)')
plt.xlabel('Number of tokens')
plt.ylabel('Frequency')
plt.show()

#%% md
# ## 2.3) Train/Validation/Test Split (70/15/15)
#%%

from sklearn.model_selection import train_test_split

X = df[['sentence','clean_text','tokens']].copy()
y = df['label'].astype(int)

# First: Train (70%) vs Temp (30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# Second: Validation (15%) vs Test (15%) by splitting temp in half
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print("Shapes:")
print("Train:", X_train.shape, y_train.shape)
print("Val:  ", X_val.shape, y_val.shape)
print("Test: ", X_test.shape, y_test.shape)

print("Saved: train_split.csv, val_split.csv, test_split.csv")

#%% md
# ## 3.1 TF-IDF features for classical ML
#%%
from sklearn.feature_extraction.text import TfidfVectorizer

# Hyperparameters for TF-IDF
max_features = 5000       # Limit vocabulary size (can tune)
ngram_range = (1, 2)      # Use unigrams and bigrams

tfidf_vectorizer = TfidfVectorizer(
    max_features=max_features,
    ngram_range=ngram_range,
    stop_words='english'  # remove common English stopwords
)

X_train_text = X_train['clean_text'].astype(str)
X_val_text   = X_val['clean_text'].astype(str)
X_test_text  = X_test['clean_text'].astype(str)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
X_val_tfidf   = tfidf_vectorizer.transform(X_val_text)
X_test_tfidf  = tfidf_vectorizer.transform(X_test_text)

print("TF-IDF train shape:", X_train_tfidf.shape)
print("TF-IDF val shape:",   X_val_tfidf.shape)
print("TF-IDF test shape:",  X_test_tfidf.shape)

#%% md
# ## 3.2 Tokenization and Padding for Deep Learning Model
#%%
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_words = 10000   # Size of the vocabulary (can tune)
max_len = 100       # Maximum sequence length (can tune)

X_train_text_dl = X_train['tokens'].apply(lambda toks: " ".join(toks))
X_val_text_dl   = X_val['tokens'].apply(lambda toks: " ".join(toks))
X_test_text_dl  = X_test['tokens'].apply(lambda toks: " ".join(toks))

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")

tokenizer.fit_on_texts(X_train_text_dl)

X_train_seq = tokenizer.texts_to_sequences(X_train_text_dl)
X_val_seq   = tokenizer.texts_to_sequences(X_val_text_dl)
X_test_seq  = tokenizer.texts_to_sequences(X_test_text_dl)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
X_val_pad   = pad_sequences(X_val_seq,   maxlen=max_len, padding='post', truncating='post')
X_test_pad  = pad_sequences(X_test_seq,  maxlen=max_len, padding='post', truncating='post')

print("Padded train shape:", X_train_pad.shape)
print("Padded val shape:",   X_val_pad.shape)
print("Padded test shape:",  X_test_pad.shape)

# For later use in the DL model:
word_index = tokenizer.word_index
print("Size of word_index (vocabulary):", len(word_index))

#%% md
# ## 4.1 Classical Model: Logistic Regression with TF-IDF
#%%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
import seaborn as sns
import matplotlib.pyplot as plt

y_train_cls = y_train
y_val_cls   = y_val
y_test_cls  = y_test

log_reg = LogisticRegression(
    max_iter=1000,
    n_jobs=-1,
    class_weight=None,   # you could set 'balanced' if classes were imbalanced
    solver='lbfgs'
)

log_reg.fit(X_train_tfidf, y_train_cls)

# Validation performance
y_val_pred = log_reg.predict(X_val_tfidf)

print("Validation classification report (Logistic Regression):")
print(classification_report(y_val_cls, y_val_pred, digits=3))

# Test performance
y_test_pred = log_reg.predict(X_test_tfidf)

print("Test classification report (Logistic Regression):")
print(classification_report(y_test_cls, y_test_pred, digits=3))

# Accuracy shortcut
test_acc = accuracy_score(y_test_cls, y_test_pred)
print(f"Test Accuracy: {test_acc:.3f}")

# Confusion matrix on test set
cm = confusion_matrix(y_test_cls, y_test_pred)
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred 0', 'Pred 1'],
            yticklabels=['True 0', 'True 1'])
plt.title('Confusion Matrix - Logistic Regression (Test)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()

#%% md
# ## 4.2 Deep Learning Model: BiLSTM
#%%
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, LSTM, Bidirectional, Dense, Dropout, SpatialDropout1D
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

# ==== Hyperparameters (more conservative) ====
embedding_dim = 50      # smaller embedding
lstm_units    = 32      # smaller LSTM
dropout_rate  = 0.5
spatial_dropout = 0.3   # dropout on embeddings
batch_size    = 64
epochs        = 20

y_train_dl = np.array(y_train)
y_val_dl   = np.array(y_val)
y_test_dl  = np.array(y_test)

vocab_size = min(max_words, len(word_index) + 1)

model = Sequential([
    # 1) Embedding + spatial dropout to regularize word embeddings
    Embedding(input_dim=vocab_size,
              output_dim=embedding_dim,
              input_length=max_len),
    SpatialDropout1D(spatial_dropout),

    # 2) Smaller BiLSTM with dropout + L2 regularization
    Bidirectional(
        LSTM(
            lstm_units,
            return_sequences=False,
            dropout=0.3,              # dropout on inputs to LSTM
            recurrent_dropout=0.3,    # dropout on recurrent connections
            kernel_regularizer=l2(1e-4),
            recurrent_regularizer=l2(1e-4)
        )
    ),

    # 3) Smaller dense layer with L2 + dropout
    Dense(32, activation='relu', kernel_regularizer=l2(1e-4)),
    Dropout(dropout_rate),

    # 4) Output layer (binary)
    Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=2,             # stop a bit earlier
    restore_best_weights=True
)

history = model.fit(
    X_train_pad, y_train_dl,
    validation_data=(X_val_pad, y_val_dl),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[early_stop],
    verbose=1
)

#%%
import matplotlib.pyplot as plt

# Plot accuracy
plt.figure()
plt.plot(history.history['accuracy'], label='Train accuracy')
plt.plot(history.history['val_accuracy'], label='Val accuracy')
plt.title('Training vs Validation Accuracy (BiLSTM)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot loss
plt.figure()
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Val loss')
plt.title('Training vs Validation Loss (BiLSTM)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

# Evaluate loss & accuracy directly
test_loss, test_acc = model.evaluate(X_test_pad, y_test_dl, verbose=0)
print(f"Test Loss (BiLSTM): {test_loss:.3f}")
print(f"Test Accuracy (BiLSTM): {test_acc:.3f}")

# Predicted probabilities and class labels
y_test_proba = model.predict(X_test_pad).ravel()
y_test_pred_dl = (y_test_proba >= 0.5).astype(int)

print("\nClassification report (BiLSTM):")
print(classification_report(y_test_dl, y_test_pred_dl, digits=3))

cm_dl = confusion_matrix(y_test_dl, y_test_pred_dl)
plt.figure(figsize=(4,3))
sns.heatmap(cm_dl, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred 0', 'Pred 1'],
            yticklabels=['True 0', 'True 1'])
plt.title('Confusion Matrix - BiLSTM (Test)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()
