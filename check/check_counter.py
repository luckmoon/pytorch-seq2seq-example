#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import Counter


def get_word_counter(texts):
    texts = [text.lower() for text in texts]
    texts = [word for text in texts for word in text.split()]
    counter = Counter(texts)
    return counter


texts_1 = ["I have a dog",
           "I also have a cat"]
texts_2 = ["You have a pig",
           "You also have a monkey"]
counter_1 = get_word_counter(texts_1)
counter_2 = get_word_counter(texts_2)
print(counter_1)
print(counter_2)
print(counter_1 + counter_2)
