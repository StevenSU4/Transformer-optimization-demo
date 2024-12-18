import torch
torch.utils.data.datapipes.utils.common.DILL_AVAILABLE = torch.utils._import_utils.dill_available()
from torch import nn
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from lion_pytorch import Lion
import matplotlib.pyplot as plt

import torchtext
torchtext.disable_torchtext_deprecation_warning()

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load IMDb dataset
import datasets
dataset = datasets.load_dataset('imdb')
train_iter = dataset['train']
test_iter = dataset['test']

# Tokenizer
tokenizer = get_tokenizer("basic_english")

# Build vocabulary
def yield_tokens(data_iter): 
    for item in data_iter: 
        yield tokenizer(item['text'])

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<pad>", "<unk>"])
vocab.set_default_index(vocab["<unk>"])  # Handle unknown tokens

# Hyperparameters
# BATCH_SIZE = 32
BATCH_SIZE = 256
MAX_SEQ_LEN = 128  # Max length of sequences

# Collate function for DataLoader
def collate_batch(batch):
    texts, labels = zip(*[(item['text'], item['label']) for item in batch]) 
    labels = torch.tensor(labels) 
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
model_for_lion = TransformerModel(len(vocab), EMBED_SIZE, NUM_HEADS, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES).to(device)
    
# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizers
sgd_lr = 0.01
optimizer_sgd = torch.optim.SGD(model_for_sgd.parameters(), lr=sgd_lr)
optimizer_adam = torch.optim.Adam(model_for_adam.parameters(), lr=0.001)
optimizer_lion = Lion(model_for_lion.parameters(), lr=1e-4)


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
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0

    def __call__(self, train_loss):
        if self.best_loss is None:
            self.best_loss = train_loss
        elif train_loss < self.best_loss - self.min_delta:
            self.best_loss = train_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

# Training process
MAX_EPOCHS = 200
PATIENCE = 10

def train_with_early_stopping(model, train_loader, test_loader, optimizer, criterion, max_epochs, patience, optimizer_name):
    early_stopping = EarlyStopping(patience=patience, min_delta=0.01)
    train_losses = []
    for epoch in range(max_epochs):
        train_loss, train_acc = train_model(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate_model(model, test_loader, criterion)
        train_losses.append(train_loss)
        print(f"Epoch {epoch+1}/{max_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {val_loss:.4f}, Test Acc: {val_acc:.4f}")
        
        if early_stopping(train_loss):
            print("Early stopping due to no significant change in training loss.")
            break

    plot_training_loss(train_losses, optimizer_name)

def plot_training_loss(train_losses, optimizer_name):
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training Loss vs Epochs for {optimizer_name}')
    plt.legend()
    plt.savefig(f'training_loss_{optimizer_name}_256.png')
    plt.close()

print(f"Training with SGD of lr={sgd_lr}.")
train_with_early_stopping(model_for_sgd, train_loader, test_loader, optimizer_sgd, criterion, MAX_EPOCHS, PATIENCE, 'SGD')

print(f"Training with Adam.")
train_with_early_stopping(model_for_adam, train_loader, test_loader, optimizer_adam, criterion, MAX_EPOCHS, PATIENCE, 'Adam')

print(f"Training with Lion.")
train_with_early_stopping(model_for_lion, train_loader, test_loader, optimizer_lion, criterion, MAX_EPOCHS, PATIENCE, 'Lion')
