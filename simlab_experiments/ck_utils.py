import random
from nltk.corpus import words

def generate_random_word():
    word_list = words.words()
    filtered_words = [word.lower() for word in word_list if len(word) == 4]
    return random.choice(filtered_words)

if __name__ == '__main__':
    print('Random word:', generate_random_word())



