import torch
import torch.nn as nn
import torch.optim as optim
from adam_mini import Adam_mini

model = nn.Linear(2, 1)

EMBED_SIZE = 64
NUM_HEADS = 4
HIDDEN_DIM = 128
NUM_LAYERS = 2
NUM_CLASSES = 2  # Binary classification

criterion = nn.MSELoss()

optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0)
optimizer = Adam_mini(
    named_parameters=model.named_parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0,
    dim=EMBED_SIZE,
    n_heads=NUM_HEADS,
    n_kv_heads=NUM_HEADS,
)


inputs = torch.tensor([[1.0, 2.0]], requires_grad=True)
target = torch.tensor([[3.0]])

output = model(inputs)
loss = criterion(output, target)

loss.backward()

optimizer.step()

optimizer.zero_grad()