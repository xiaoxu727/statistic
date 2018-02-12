# 随机采样和排序
import pandas as pd
import numpy as np
from pandas import DataFrame, Series


def get_card():
    car_val = ((list(range(1, 11)) + [10] * 3))*4
    suits = ['H', 'S', 'C', 'D']
    base_name = ['A'] + list(range(2, 11)) + ['J', 'Q', 'K']
    cards = []
    for i in suits:
        cards.extend(str(s) + i for s in base_name)
    # print(cards)
    deck = Series(car_val, index=cards)
    # print(deck[:13])
    return deck


def draw(deck, n=5):
    return deck.take(np.random.permutation(len(deck))[:n])

if __name__ == '__main__':
    deck = get_card()
    print(draw(deck))
    get_suit = lambda card: card[-1]
    print(deck.groupby(get_suit).apply(draw, n=2))
    print(deck.groupby(get_suit, group_keys=False).apply(draw, n=2))


