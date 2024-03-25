import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

# Загрузка модели
train_on_gpu = torch.cuda.is_available()
def one_hot_encode(arr, n_labels):

    # Initialize the the encoded array
    one_hot = np.zeros((arr.size, n_labels), dtype=np.float32)

    # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.0

    # Finally reshape it to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))

    return one_hot
def predict(net, char, h=None, top_k=None):
    """Given a character, predict the next character.
    Returns the predicted character and the hidden state.
    """

    # tensor inputs
    print(char)
    x = np.array([[net.char2int[char]]])
    x = one_hot_encode(x, len(net.chars))
    inputs = torch.from_numpy(x)

    if train_on_gpu:
        inputs = inputs.cuda()

    # detach hidden state from history
    h = tuple([each.data for each in h])
    # get the output of the model
    out, h = net(inputs, h)

    # get the character probabilities
    p = F.softmax(out, dim=1).data
    if train_on_gpu:
        p = p.cpu()  # move to cpu

    # get top characters
    if top_k is None:
        top_ch = np.arange(len(net.chars))
    else:
        p, top_ch = p.topk(top_k)
        top_ch = top_ch.numpy().squeeze()

    # select the likely next character with some element of randomness
    p = p.numpy().squeeze()
    char = np.random.choice(top_ch, p=p / p.sum())

    # return the encoded value of the predicted char and the hidden state
    return net.int2char[char], h

class CharRNN(nn.Module):
    def __init__(self, tokens, n_hidden=256, n_layers=2, drop_prob=0.5, lr=0.001):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr

        # creating character dictionaries
        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}

        ## TODO: define the LSTM
        self.lstm = nn.LSTM(
            len(self.chars), n_hidden, n_layers, dropout=drop_prob, batch_first=True
        )

        ## TODO: define a dropout layer
        self.dropout = nn.Dropout(drop_prob)

        ## TODO: define the final, fully-connected output layer
        self.fc = nn.Linear(n_hidden, len(self.chars))

    def forward(self, x, hidden):
        """Forward pass through the network.
        These inputs are x, and the hidden/cell state `hidden`."""
        r_output, hidden = self.lstm(x, hidden)
        out = self.dropout(r_output)
        out = out.contiguous().reshape(-1, self.n_hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        """Initializes hidden state"""
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        if train_on_gpu:
            hidden = (
                weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
            )
        else:
            hidden = (
                weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
            )
        return hidden

def sample(net, size, prime="The", top_k=None):

    if train_on_gpu:
        net.cuda()
    else:
        net.cpu()
    net.eval()  # eval mode
    total_params = sum(p.numel() for p in net.parameters())
    chars = [ch for ch in prime]
    h = net.init_hidden(1)
    for ch in prime:
        char, h = predict(net, ch, h, top_k=top_k)
    chars.append(char)

    for _ in range(size):
        char, h = predict(net, chars[-1], h, top_k=top_k)
        chars.append(char)

    return "".join(chars)
def  predictText(text):

    model_name = "rnn_x_epoch.net"
    checkpoint = torch.load(model_name)
    tokens = checkpoint["tokens"]
    model = CharRNN(tokens=tokens, n_hidden=checkpoint["n_hidden"], n_layers=checkpoint["n_layers"])
    model.load_state_dict(checkpoint["state_dict"])

    # model.eval()
    out2=sample(model, 1000, prime=text, top_k=5)
    return out2
