from collections import Counter
import pandas as pd
import numpy as np
from math import ceil
import torch
from sklearn.utils import shuffle
from .model_construction import LSTM_architecture


def read_data():
    n = ['text']
    data_positive_current = pd.read_csv('nlp/combined_output_positive.csv', sep=';', names=n, usecols=['text'])
    total_rows = len(data_positive_current)
    sample_size = total_rows // 2
    reduced_data = data_positive_current.sample(n=sample_size, random_state=42)  # random_state для воспроизводимости
    reduced_data.to_csv('combined_output_positive_double.csv', sep=';', index=False, header=False)
    data_positive_current= pd.read_csv('combined_output_positive_double.csv', sep=';', names=n, usecols=['text'])


    data_positive_new = pd.read_csv('new_positive_reviews.csv', sep=';', names=n, usecols=['text'])
    data_positive = pd.concat([data_positive_current, data_positive_new], ignore_index=True)

    data_negative_current = pd.read_csv('combined_output_negative.csv', sep=';', names=n, usecols=['text'])
    data_negative_new = pd.read_csv('new_negative_reviews.csv', sep=';', names=n, usecols=['text'])
    data_negative = pd.concat([data_negative_current, data_negative_new], ignore_index=True)

    ### Формирование сбалансированного датасета
    sample_size = 40000
    reviews_withoutshuffle = np.concatenate((data_positive['text'].values[:sample_size],
                                             data_negative['text'].values[:sample_size]), axis=0)
    labels_withoutshuffle = np.asarray([1] * sample_size + [0] * sample_size)

    assert len(reviews_withoutshuffle) == len(labels_withoutshuffle)
    texts, labels = shuffle(reviews_withoutshuffle, labels_withoutshuffle, random_state=0)

    return  texts, labels

texts, labels = read_data()

#'"#$%&\'()*+-/:;<=>[\]^_`{|}~'

def tokenize():
    punctuation = '"#$%&\'()*+-/:;<=>[\]^_`{|}~'
    all_texts = 'separator'.join(texts)
    all_texts = all_texts.lower()
    all_text = ''.join([c for c in all_texts if c not in punctuation])
    texts_split = all_text.split('separator')
    all_text = ' '.join(texts_split)
    words = all_text.split()
    return words


def get_vocabulary():
    words = tokenize()
    counts = Counter(words)
    vocab = sorted(counts, key=counts.get, reverse=True)
    vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}
    return vocab, vocab_to_int


def tokenize_text(test_review):
    test_review = test_review.lower()
    punctuation = '"#$%&\'()*+-/:;<=>[\]^_`{|}~'
    test_text = ''.join([c for c in test_review if c not in punctuation])
    test_words = test_text.split()

    new_text = []
    for word in test_words:
        if (word[0] != '@') & ('http' not in word) & (~word.isdigit()):
            new_text.append(word)
    test_ints = []
    test_ints.append([vocab_to_int[word] for word in new_text])

    return test_ints


def add_pads(texts_ints, seq_length):
    features = np.zeros((len(texts_ints), seq_length), dtype=int)

    for i, row in enumerate(texts_ints):
        if len(row) == 0:
            continue
        features[i, -len(row):] = np.array(row)[:seq_length]

    return features


def predict(net, test_text, sequence_length=30):
    net.eval()
    test_ints = tokenize_text(test_text)
    seq_length = sequence_length
    features = add_pads(test_ints, seq_length)
    feature_tensor = torch.from_numpy(features)

    batch_size = feature_tensor.size(0)
    h = net.init_hidden_state(batch_size)
    output, h = net(feature_tensor, h)

    pred = torch.round(output.squeeze())
    pos_prob = output.item()

    if (pred.item() == 1):
        result = "Позитивное сообщение"
    else:
        result = "Негативное сообщение"

    return result, pos_prob


def scan(string):

    input_text = string
    _, vocab_to_int = get_vocabulary()


    vocab_size = len(vocab_to_int)+1
    print(f'Vocab size: {vocab_size}')
    output_size = 1
    embedding_dim = 100
    hidden_dim = 128
    number_of_layers = 2
    model = LSTM_architecture(vocab_size, output_size, embedding_dim, hidden_dim, number_of_layers)
    model.load_state_dict(torch.load("model_from_colab (1).pt", weights_only=True))
    model.eval()
    seq_length = 30

    type_of_tonal, pos_prob = predict(model, input_text, seq_length)

    print("Окраска - {}, вероятность = {}%".format(type_of_tonal, 0))

    if type_of_tonal == "Негативное сообщение":
        prob = ceil((1 - pos_prob) * 100)
        return "Не рекомендую"
    else:
        prob = ceil(pos_prob * 100)
        return "Рекомендую"

if __name__ == "__main__":
    _, vocab_to_int = get_vocabulary()
    vocab_size = len(vocab_to_int) + 1
    output_size = 1
    embedding_dim = 200
    hidden_dim = 128
    number_of_layers = 2
    model = LSTM_architecture(vocab_size, output_size, embedding_dim, hidden_dim, number_of_layers)
    model.load_state_dict(torch.load("best_model.pt", map_location=torch.device('cpu')))
    seq_length = 30

    test_review1 = "не понравилось"
    test_review2 = "Фильм ужасен! Никогда больше не буду смотреть!"
    test_review3 = "Нормальный фильм. Ничего особенного"
    test_review4 = "Крутой"
    test_review5 = "Не крутой"
    test_review6 = "Понравился"
    test_review7 = "Не понравился"
    test_review8 = "Рекомендую"

    print(f"Модель на устройстве: {next(model.parameters()).device}")
    type_of_tonal1, _ = predict(model, test_review1, seq_length)
    print("Окраска - {}".format(type_of_tonal1))
    type_of_tonal2, _ = predict(model, test_review2, seq_length)
    print("Окраска - {}".format(type_of_tonal1))
    type_of_tonal3, _ = predict(model, test_review3, seq_length)
    print("Окраска - {}".format(type_of_tonal1))
    type_of_tonal4, _ = predict(model, test_review4, seq_length)
    print("Окраска - {}".format(type_of_tonal1))
    type_of_tonal5, _ = predict(model, test_review5, seq_length)
    print("Окраска - {}".format(type_of_tonal1))
    type_of_tonal6, _ = predict(model, test_review6, seq_length)
    print("Окраска - {}".format(type_of_tonal1))
    type_of_tonal7, _ = predict(model, test_review7, seq_length)
    print("Окраска - {}".format(type_of_tonal1))
    type_of_tonal8, _ = predict(model, test_review8, seq_length)
    print("Окраска - {}".format(type_of_tonal1))
#Vocab size: 345796
# Негативное сообщение
# Позитивное сообщение
# Негативное сообщение
# Позитивное сообщение
# Негативное сообщение
# Позитивное сообщение
# Позитивное сообщение
# Позитивное сообщение