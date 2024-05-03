import torch

import torch.nn as nn


class SimpleMLPAdapter(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleMLPAdapter, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size)
        self.relu = nn.ReLU()

    def forward(self, image_embeddings, prompt_embeddings):
        inputs = torch.cat((image_embeddings, prompt_embeddings), dim=1)

        x = self.fc1(inputs)
        x = self.relu(x)
        x = self.fc2(x)

        return x
