import pickle
from khmernltk import word_tokenize


class KhmerNgramPredictor:

    def __init__(self, model_dir):

        self.models = {}

        for n in [2,3,4]:

            with open(f"{model_dir}/khmer_ngram_{n}.pkl","rb") as f:
                self.models[n] = pickle.load(f)


    def predict(self, text, top_k=5):

        tokens = word_tokenize(text)

        if len(tokens) >= 3:
            context = tuple(tokens[-3:])
            model = self.models[4]

        elif len(tokens) >= 2:
            context = tuple(tokens[-2:])
            model = self.models[3]

        elif len(tokens) >= 1:
            context = tuple(tokens[-1:])
            model = self.models[2]

        else:
            return []

        candidates = model.get(context, {})

        total = sum(candidates.values())

        results = []

        for word, count in candidates.most_common(top_k):

            prob = count / total

            results.append({
                "word": word,
                "score": round(prob, 4)
            })

        return results