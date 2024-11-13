import re
from collections import defaultdict
import math
import pandas as pd

class NaiveBayesClassifier:
    def __init__(self):
        self.spam_word_counts = defaultdict(int)
        self.ham_word_counts = defaultdict(int)
        self.spam_messages = 0
        self.ham_messages = 0
        self.total_spam_words = 0
        self.total_ham_words = 0

    def preprocess(self, text):
        # Преобразуем текст в нижний регистр и удаляем лишние символы
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        return words

    def train(self, messages, labels):
        for message, label in zip(messages, labels):
            words = self.preprocess(message)
            if label == 'spam':  # Спам
                self.spam_messages += 1
                for word in words:
                    self.spam_word_counts[word] += 1
                    self.total_spam_words += 1
            else:  # Не спам
                self.ham_messages += 1
                for word in words:
                    self.ham_word_counts[word] += 1
                    self.total_ham_words += 1

    def predict(self, message):
        words = self.preprocess(message)
        # Вероятность спама и не спама
        spam_prob = math.log(self.spam_messages / (self.spam_messages + self.ham_messages))
        ham_prob = math.log(self.ham_messages / (self.spam_messages + self.ham_messages))

        for word in words:
            # Вероятности для спама
            word_spam_prob = (self.spam_word_counts[word] + 1) / (self.total_spam_words + len(self.spam_word_counts))
            spam_prob += math.log(word_spam_prob)
            # Вероятности для не спама
            word_ham_prob = (self.ham_word_counts[word] + 1) / (self.total_ham_words + len(self.ham_word_counts))
            ham_prob += math.log(word_ham_prob)

        # Возвращаем метку на основе максимальной вероятности
        return 'spam' if spam_prob > ham_prob else 'ham'


# Загрузка и подготовка данных
df = pd.read_csv('./spam.csv', usecols=[0, 1], encoding='ISO-8859-1')
df.columns = ['label', 'message']

# Разделяем текст и метки на отдельные списки
texts = df['message'].tolist()
labels = df['label'].tolist()

# Создаем и обучаем классификатор
classifier = NaiveBayesClassifier()
classifier.train(texts, labels)

# Пример данных для проверки
test_data = [
    "Congratulations! You've been selected to win a $1000 gift card. Claim it now!",
    "Hey, are you free this weekend? Let's catch up!",
    "You've won a cash prize! Click here to claim it now.",
    "Don't forget our meeting tomorrow at 10 AM.",
    "Limited time offer: 50% off on all items! Hurry up!",
    "Can you send me the report by end of the day?",
    "Exclusive offer just for you! Grab your free membership now.",
    "Your bank statement is ready. Please log in to your account to view it.",
    "Win a million dollars by just clicking on this link!",
    "Let's meet at the cafe tomorrow afternoon. I'll be waiting.",
]


# Предсказание
for message in test_data:
    prediction = classifier.predict(message)
    print(f'Text: "{message}" -> Prediction: {"Spam" if prediction == "spam" else "Not Spam"}')
