class CharTokenizer:
    def __init__(self):
        vocab = (
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789"
            " .,;!?()[]{}<>:\"'\\/@#$%^&*-+=\n"
        )

        self.chars = sorted(list(set(vocab)))
        self.vocab_size = len(self.chars)

        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text):
        return [self.stoi.get(c, self.stoi[" "]) for c in text]

    def decode(self, ids):
        return ''.join([self.itos.get(i, " ") for i in ids])
