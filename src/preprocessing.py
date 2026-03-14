import re
import pickle
import math
from collections import defaultdict, Counter
from khmernltk import word_tokenize


KHMER_PUNCT = ["។","៕","៖","ៗ","៘","៙",",",".","!","?",";",":","(",")","\"","'"]


def clean_tokens(tokens):

    cleaned = []

    for token in tokens:

        token = token.strip()

        if token == "":
            continue

        if token in KHMER_PUNCT:
            continue

        if re.match(r'^[a-zA-Z]+$', token):
            continue

        if re.match(r'^[0-9]+$', token):
            continue

        cleaned.append(token)

    return cleaned


def tokenize_khmer(text):

    tokens = word_tokenize(text)

    tokens = clean_tokens(tokens)

    return tokens


def build_ngram(tokens, n):

    model = defaultdict(Counter)

    for i in range(len(tokens) - n + 1):

        context = tuple(tokens[i:i+n-1])
        target = tokens[i+n-1]

        model[context][target] += 1

    return model


def save_model(model, path):

    with open(path, "wb") as f:
        pickle.dump(model, f)


def perplexity(tokens, model, n):

    N = len(tokens)
    log_prob = 0
    V = len(set(tokens))

    for i in range(n-1, N):

        context = tuple(tokens[i-n+1:i])
        word = tokens[i]

        count = model[context][word] + 1
        total = sum(model[context].values()) + V

        prob = count / total

        log_prob += math.log(prob)

    return math.exp(-log_prob / N)


def train_models(data_path, save_dir):

    models = {
        2: defaultdict(Counter),
        3: defaultdict(Counter),
        4: defaultdict(Counter)
    }

    token_buffer = []

    with open(data_path, "r", encoding="utf-8") as f:

        for line in f:

            tokens = tokenize_khmer(line)

            token_buffer.extend(tokens)

            # process when buffer large
            if len(token_buffer) > 5000:

                update_models(token_buffer, models)

                token_buffer = token_buffer[-4:]  # keep small context

    # process remaining tokens
    if token_buffer:
        update_models(token_buffer, models)

    # save models
    for n in models:

        save_model(models[n], f"{save_dir}/khmer_ngram_{n}.pkl")

    print("Training complete")
    
def update_models(tokens, models):

    for n in models:

        model = models[n]

        for i in range(len(tokens) - n + 1):

            context = tuple(tokens[i:i+n-1])
            target = tokens[i+n-1]

            model[context][target] += 1