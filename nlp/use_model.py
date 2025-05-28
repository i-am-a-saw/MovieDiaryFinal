from collections import Counter
import pandas as pd
import numpy as np
from math import ceil
import torch
from sklearn.utils import shuffle
from .model_construction import LSTM_architecture
import pickle

with open('vocab_to_int.pkl', 'rb') as f:
    vocab_to_int = pickle.load(f)
vocab = list(vocab_to_int.keys())

def tokenize_text(test_review):
    test_review = test_review.lower()
    punctuation_to_remove = '"#$%&\'()*+-/:;<=>[\]^_`{|}~'  # Удаляем всё, кроме !, ?, .
    test_text = ''.join([c for c in test_review if c not in punctuation_to_remove])
    test_words = test_text.split()
    new_text = []
    for word in test_words:
        if (word[0] != '@') and ('http' not in word) and (not word.isdigit()):
            new_text.append(word)
    test_ints = [[vocab_to_int.get(word, 0) for word in new_text]]  # Используем .get() для обработки неизвестных слов
    return test_ints


def add_pads(texts_ints, seq_length):
    features = np.zeros((len(texts_ints), seq_length), dtype=int)

    for i, row in enumerate(texts_ints):
        if len(row) == 0:
            continue
        features[i, -len(row):] = np.array(row)[:seq_length]

    return features


def predict(net, test_text, sequence_length=25):
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
    vocab_size = len(vocab_to_int) + 1
    output_size = 1
    embedding_dim = 100
    hidden_dim = 128
    number_of_layers = 2
    model = LSTM_architecture(vocab_size, output_size, embedding_dim, hidden_dim, number_of_layers)
    model.load_state_dict(torch.load("model_check_last_new.pt", map_location=torch.device('cpu')))
    seq_length = 25

    test_review1 = "развязка интересная, режиссер постарался"


    print(f"Модель на устройстве: {next(model.parameters()).device}")
    type_of_tonal1, _ = predict(model, test_review1, seq_length)
    print("Окраска - {}".format(type_of_tonal1))

#Vocab size: 345796
# Негативное сообщение
# Позитивное сообщение
# Негативное сообщение
# Позитивное сообщение
# Позитивное сообщение
# Позитивное сообщение
# Позитивное сообщение
# Позитивное сообщение
