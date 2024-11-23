import torch
import torch.nn as nn
import torch.optim as optim
from adam_mini import Adam_mini

model = nn.Linear(2, 1)

criterion = nn.MSELoss()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0)
optimizer = Adam_mini(
    named_parameters=model.named_parameters(),
    lr=lr,
    betas=(beta1, beta2),
    eps=eps,
    weight_decay=weight_decay,
    # dim=model_config.dim,
    # n_heads=model_config.n_heads,
    # n_kv_heads=model_config.n_kv_heads,
)


inputs = torch.tensor([[1.0, 2.0]], requires_grad=True)
target = torch.tensor([[3.0]])

output = model(inputs)
loss = criterion(output, target)

loss.backward()

optimizer.step()

optimizer.zero_grad()