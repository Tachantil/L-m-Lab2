import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import defaultdict, Counter
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')

pos_data = open("positive-words.txt").read().splitlines()
neg_data = open("negative-words.txt").read().splitlines()

def clean_text(text):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    tokens = word_tokenize(text.lower())
    return [stemmer.stem(word) for word in tokens if word.isalpha() and word not in stop_words]

pos_data = [clean_text(tweet) for tweet in pos_data]
neg_data = [clean_text(tweet) for tweet in neg_data]

def build_freq_dict(pos_data, neg_data):
    freq_map = defaultdict(lambda: [0, 0])
    for tweet in pos_data:
        for token in tweet:
            freq_map[token][0] += 1
    for tweet in neg_data:
        for token in tweet:
            freq_map[token][1] += 1
    return freq_map

freq_map = build_freq_dict(pos_data, neg_data)

prior_log = np.log(len(pos_data) / len(neg_data))

def calc_log_like(freq_map, pos_data, neg_data):
    vocab_size = len(freq_map)
    pos_total = sum([freq[0] for freq in freq_map.values()])
    neg_total = sum([freq[1] for freq in freq_map.values()])

    log_likes = {}
    for token, (pos_count, neg_count) in freq_map.items():
        p_pos = (pos_count + 1) / (pos_total + vocab_size)
        p_neg = (neg_count + 1) / (neg_total + vocab_size)
        log_likes[token] = np.log(p_pos / p_neg)
    return log_likes

log_likes = calc_log_like(freq_map, pos_data, neg_data)

def bayes_predict(tweet, prior_log, log_likes):
    tokens = clean_text(tweet)
    score = prior_log
    for token in tokens:
        if token in log_likes:
            score += log_likes[token]
    return "Позитивний" if score > 0 else "Негативний"

test_data = [
    ("I love this product!", "Позитивний"),
    ("This is the worst thing ever!", "Негативний"),
    ("Amazing service, I am very happy.", "Позитивний"),
    ("I hate waiting for so long.", "Негативний"),
]

correct_preds = 0
for tweet, label in test_data:
    pred = bayes_predict(tweet, prior_log, log_likes)
    print(f"Твіт: \"{tweet}\" | Передбачення: {pred} | Правильна відповідь: {label}")
    if pred == label:
        correct_preds += 1

accuracy = correct_preds / len(test_data)
print(f"\nТочність моделі: {accuracy:.2f}")

top_tokens = sorted(log_likes.items(), key=lambda x: x[1], reverse=True)
print("\nТОП-5 найпозитивніших токенів:")
for token, score in top_tokens[:5]:
    print(f"{token}: {score:.2f}")
    
print("\nТОП-5 найнегативніших токенів:")
for token, score in top_tokens[-5:]:
    print(f"{token}: {score:.2f}")

print("\nКастомний тест:")
my_tweet = "I am excited about the new project, but worried about the deadline."
print(f"Твіт: \"{my_tweet}\" | Передбачення: {bayes_predict(my_tweet, prior_log, log_likes)}")
