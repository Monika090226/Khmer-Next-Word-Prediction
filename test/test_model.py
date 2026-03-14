import sys
import os

sys.path.append(os.path.abspath(".."))

from src.inference import KhmerNgramPredictor


predictor = KhmerNgramPredictor("../model")


text = "សាលារាជធានីភ្នំពេញ"

pred = predictor.predict(text)

print("Input:", text)
print("Suggestions:", pred)