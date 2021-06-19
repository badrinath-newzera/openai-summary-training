import pickle
import numpy as np

with open('/home/badri/mansion/openai-training/lm-human-preferences/tmp/pickle_fn', 'rb') as file:
    k = pickle.load(file)

print(k)

