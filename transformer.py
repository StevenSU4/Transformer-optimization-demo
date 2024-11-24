import torch
torch.utils.data.datapipes.utils.common.DILL_AVAILABLE = torch.utils._import_utils.dill_available()
from torch import nn
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from lion_pytorch import Lion
import torch.nn.functional as F
from sophia import SophiaG

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
BATCH_SIZE = 32
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
    def __init__(self, vocab_size, embed_size, num_heads, hidden_dim, num_layers, num_classes, dropout=0.3):
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
model_for_sophia = TransformerModel(len(vocab), EMBED_SIZE, NUM_HEADS, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES).to(device)
    
# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizers
sgd_lr = 0.1
optimizer_sgd = torch.optim.SGD(model_for_sgd.parameters(), lr=sgd_lr)
optimizer_adam = torch.optim.Adam(model_for_adam.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=2e-3)
optimizer_lion = Lion(model_for_lion.parameters(), lr=2e-4, weight_decay=1e-2)
optimizer_sophia = SophiaG(model_for_sophia.parameters(), lr=8e-4, betas=(0.9, 0.999), weight_decay=4e-3)

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
# MAX_EPOCHS = 50

# print(f"Training with SGD of lr={sgd_lr}.")
# for epoch in range(EPOCHS):
#     train_loss, train_acc = train_model(model_for_sgd, train_loader, optimizer_sgd, criterion)
#     test_loss, test_acc = evaluate_model(model_for_sgd, test_loader, criterion)
#     print(f"Epoch {epoch+1}/{EPOCHS}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
# print(f"Training with Adam.")
# for epoch in range(EPOCHS):
#     train_loss, train_acc = train_model(model_for_adam, train_loader, optimizer_adam, criterion)
#     test_loss, test_acc = evaluate_model(model_for_adam, test_loader, criterion)
#     print(f"Epoch {epoch+1}/{EPOCHS}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

# print(f"Training with Lion.")
# for epoch in range(EPOCHS):
#     train_loss, train_acc = train_model(model_for_lion, train_loader, optimizer_lion, criterion)
#     test_loss, test_acc = evaluate_model(model_for_lion, test_loader, criterion)
#     print(f"Epoch {epoch+1}/{EPOCHS}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")


# Training process
MAX_EPOCHS = 50

def train_with_early_stopping(model, train_loader, test_loader, optimizer, criterion, max_epochs):
    for epoch in range(max_epochs):
        train_loss, train_acc = train_model(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate_model(model, test_loader, criterion)
        print(f"Epoch {epoch+1}/{max_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {val_loss:.4f}, Test Acc: {val_acc:.4f}")
        
        if train_loss < 0.1:
            print("Already converged.")
            break

print(f"Training with SGD of lr={sgd_lr}.")
train_with_early_stopping(model_for_sgd, train_loader, test_loader, optimizer_sgd, criterion, MAX_EPOCHS)

print(f"Training with Adam.")
train_with_early_stopping(model_for_adam, train_loader, test_loader, optimizer_adam, criterion, MAX_EPOCHS)

print(f"Training with Lion.")
train_with_early_stopping(model_for_lion, train_loader, test_loader, optimizer_lion, criterion, MAX_EPOCHS)

print(f"Training with Sophia.")

block_size = 1  # Adjust based on your data loader
k = 10
iter_num = -1

def train_with_sophia(model, train_loader, optimizer, criterion, max_epochs, block_size, k):
    total_bs = len(train_loader) * block_size
    for epoch in range(max_epochs):
        model.train()
        total_loss, total_acc = 0, 0
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)

            # Forward pass
            outputs = model(texts)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step(bs=total_bs)
            optimizer.zero_grad(set_to_none=True)
            iter_num += 1

            # Metrics
            total_loss += loss.item()
            total_acc += (outputs.argmax(dim=1) == labels).sum().item()

            if iter_num % k != k - 1:
                continue
            else:
                # Update Hessian EMA
                outputs = model(texts)
                samp_dist = torch.distributions.Categorical(logits=outputs)
                y_sample = samp_dist.sample()
                loss_sampled = F.cross_entropy(outputs.view(-1, outputs.size(-1)), y_sample.view(-1), ignore_index=-1)
                loss_sampled.backward()
                optimizer.update_hessian()
                optimizer.zero_grad(set_to_none=True)
                model.zero_grad()

        print(f"Epoch {epoch+1}/{max_epochs}: Train Loss: {total_loss / len(train_loader):.4f}, Train Acc: {total_acc / len(train_loader.dataset):.4f}")
        
        if (total_loss / len(train_loader)) < 0.1:
            print("Already converged.")
            break

train_with_sophia(model_for_sophia, train_loader, optimizer_sophia, criterion, MAX_EPOCHS, block_size, k)