import torch
torch.utils.data.datapipes.utils.common.DILL_AVAILABLE = torch.utils._import_utils.dill_available()
from torch import nn
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from adam_mini import Adam_mini

import torchtext
torchtext.disable_torchtext_deprecation_warning()

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load IMDb dataset
# from torchtext.datasets import IMDB
# train_iter, test_iter = IMDB(split=('train', 'test'))
import datasets
dataset = datasets.load_dataset('imdb')
train_iter = dataset['train']
test_iter = dataset['test']

# Tokenizer
tokenizer = get_tokenizer("basic_english")

# Build vocabulary
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<pad>", "<unk>"])
vocab.set_default_index(vocab["<unk>"])  # Handle unknown tokens

# Hyperparameters
BATCH_SIZE = 32
MAX_SEQ_LEN = 128  # Max length of sequences

# Collate function for DataLoader
def collate_batch(batch):
    labels, texts = zip(*batch)
    labels = torch.tensor([1 if label == 'pos' else 0 for label in labels])
    tokenized_texts = [torch.tensor(vocab(tokenizer(text)))[:MAX_SEQ_LEN] for text in texts]
    padded_texts = nn.utils.rnn.pad_sequence(tokenized_texts, batch_first=True, padding_value=vocab["<pad>"])
    return padded_texts, labels

# Dataloaders
train_loader = DataLoader(list(train_iter), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(list(test_iter), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, hidden_dim, num_layers, num_classes, dropout=0.2):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, MAX_SEQ_LEN, embed_size))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_size, num_classes)

    def forward(self, x):
        # Add positional encoding to embeddings
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        x = self.transformer(x.permute(1, 0, 2))  # [Seq, Batch, Embed]
        x = x.mean(dim=0)  # Pooling over sequence
        x = self.dropout(x)
        return self.fc(x)

# Initialize model
EMBED_SIZE = 32
NUM_HEADS = 2
HIDDEN_DIM = 64
NUM_LAYERS = 1
NUM_CLASSES = 2  # Binary classification

model_for_sgd = TransformerModel(len(vocab), EMBED_SIZE, NUM_HEADS, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES).to(device)
model_for_adam = TransformerModel(len(vocab), EMBED_SIZE, NUM_HEADS, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES).to(device)

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizers
optimizer_sgd = torch.optim.SGD(model_for_sgd.parameters(), lr=0.01)
optimizer_adam = torch.optim.Adam(model_for_adam.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0)
# optimize_adam_mini = Adam_mini(
#     named_parameters=model.named_parameters(),
#     lr=0.001,
#     betas=(0.9, 0.999),
#     eps=1e-8,
#     weight_decay=0,
#     dim=EMBED_SIZE,
#     n_heads=NUM_HEADS,
#     n_kv_heads=NUM_HEADS,)

def train_model(model, dataloader, optimizer, criterion):
    model.train()
    total_loss, total_acc = 0, 0
    for texts, labels in dataloader:
        texts, labels = texts.to(device), labels.to(device)

        # Forward pass
        outputs = model(texts)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        total_acc += (outputs.argmax(dim=1) == labels).sum().item()

    return total_loss / len(dataloader), total_acc / len(dataloader.dataset)

def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss, total_acc = 0, 0
    with torch.no_grad():
        for texts, labels in dataloader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_acc += (outputs.argmax(dim=1) == labels).sum().item()
    return total_loss / len(dataloader), total_acc / len(dataloader.dataset)

# Training process
EPOCHS = 3
# for optimizer in [optimizer_sgd, optimizer_adam]:
#     print(f"Training with {optimizer.__class__.__name__}")
#     for epoch in range(EPOCHS):
#         train_loss, train_acc = train_model(model, train_loader, optimizer, criterion)
#         test_loss, test_acc = evaluate_model(model, test_loader, criterion)
#         print(f"Epoch {epoch+1}/{EPOCHS}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

print(f"Training with SGD.")
for epoch in range(EPOCHS):
    train_loss, train_acc = train_model(model_for_sgd, train_loader, optimizer_sgd, criterion)
    test_loss, test_acc = evaluate_model(model_for_sgd, test_loader, criterion)
    print(f"Epoch {epoch+1}/{EPOCHS}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
print(f"Training with Adam.")
for epoch in range(EPOCHS):
    train_loss, train_acc = train_model(model_for_adam, train_loader, optimizer_adam, criterion)
    test_loss, test_acc = evaluate_model(model_for_adam, test_loader, criterion)
    print(f"Epoch {epoch+1}/{EPOCHS}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
