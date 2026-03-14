import pickle
from .preprocessing import tokenize_khmer, perplexity

def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def evaluate(data_path, model_dir):

    models = {}
    for n in [2,3,4]:
        model_file = f"{model_dir}/khmer_ngram_{n}.pkl"
        print("Loading model:", model_file)
        models[n] = load_model(model_file)

    total_tokens = []
    
    # read file gradually
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = tokenize_khmer(line.strip())
            total_tokens.extend(tokens)

    print("Total tokens:", len(total_tokens))

    for n in [2,3,4]:
        ppl = perplexity(total_tokens, models[n], n)
        print(f"{n}-gram perplexity:", ppl)


if __name__ == "__main__":
    data_path="data/general-text.txt"
    model_dir="model"
    evaluate(data_path, model_dir)