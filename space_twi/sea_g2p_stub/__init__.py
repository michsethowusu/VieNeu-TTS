class SEAPipeline:
    def __init__(self, *a, **kw): pass
    def run(self, text): return text

class G2P:
    def __init__(self, *a, **kw): pass
    def phonemize_batch(self, texts, **kw): return list(texts)

class Normalizer:
    def normalize(self, text): return text
