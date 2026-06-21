import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(output_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def step(self, x, h_prev):
        h_next = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h_prev) + self.bh)
        y = np.dot(self.Why, h_next) + self.by
        return y, h_next

    def forward(self, inputs):
        h = np.zeros((self.hidden_size, 1))
        outputs = []
        for x in inputs:
            x = x.reshape(-1, 1)
            y, h = self.step(x, h)
            outputs.append(y)
        return outputs


if __name__ == "__main__":
    seq_length = 5
    input_size = 3
    hidden_size = 4
    output_size = 2

    rnn = SimpleRNN(input_size, hidden_size, output_size)
    sample_input = [np.random.randn(input_size) for _ in range(seq_length)]

    outputs = rnn.forward(sample_input)
    for t, out in enumerate(outputs, 1):
        print(f"Step {t}: output = {out.ravel()}")

