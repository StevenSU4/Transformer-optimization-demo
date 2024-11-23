import torch
torch.utils.data.datapipes.utils.common.DILL_AVAILABLE = torch.utils._import_utils.dill_available()
from torch import nn
from torch.utils.data import DataLoader
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

import torchtext
torchtext.disable_torchtext_deprecation_warning()

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load IMDb dataset
train_iter, test_iter = IMDB(split=('train', 'test'))

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
train_loader = DataLoader(list(IMDB(split='train')), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(list(IMDB(split='test')), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, hidden_dim, num_layers, num_classes, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, MAX_SEQ_LEN, embed_size))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_size, num_classes)

    def forward(self, x):
        # Add positional encoding to embeddings
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        x = self.transformer(x.permute(1, 0, 2))  # [Seq, Batch, Embed]
        x = x.mean(dim=0)  # Pooling over sequence
        return self.fc(x)

# Initialize model
EMBED_SIZE = 64
NUM_HEADS = 4
HIDDEN_DIM = 128
NUM_LAYERS = 2
NUM_CLASSES = 2  # Binary classification

model = TransformerModel(len(vocab), EMBED_SIZE, NUM_HEADS, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES).to(device)

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizers
optimizer_sgd = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer_adam = torch.optim.Adam(model.parameters(), lr=0.001)

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
EPOCHS = 5
for optimizer in [optimizer_sgd, optimizer_adam]:
    print(f"Training with {optimizer.__class__.__name__}")
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_model(model, train_loader, optimizer, criterion)
        test_loss, test_acc = evaluate_model(model, test_loader, criterion)
        print(f"Epoch {epoch+1}/{EPOCHS}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
