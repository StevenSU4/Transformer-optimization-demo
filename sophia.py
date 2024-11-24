from sophia import SophiaG
optimizer_sophia = SophiaG(
    model_for_adam_mini.parameters(),
    lr=2e-4,
    betas=(0.965, 0.99),
    rho=0.01,
    weight_decay=1e-1
)

def train_model_with_sophia(model, dataloader, optimizer, criterion, k=10):
    model.train()
    total_loss, total_acc = 0, 0
    iter_num = -1
    total_bs = len(dataloader)
    block_size = BATCH_SIZE
    bs = total_bs * block_size

    for texts, labels in dataloader:
        texts, labels = texts.to(device), labels.to(device)

        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step(bs=bs)
        optimizer.zero_grad(set_to_none=True)
        iter_num += 1

        if iter_num % k == k - 1:
            with torch.no_grad():
                logits = model(texts)
                samp_dist = torch.distributions.Categorical(logits=logits)
                y_sample = samp_dist.sample()
                loss_sampled = F.cross_entropy(logits.view(-1, logits.size(-1)), y_sample.view(-1), ignore_index=-1)
                loss_sampled.backward()
                optimizer.update_hessian()
                optimizer.zero_grad(set_to_none=True)
                model.zero_grad()

        total_loss += loss.item()
        total_acc += (outputs.argmax(dim=1) == labels).sum().item()

    return total_loss / len(dataloader), total_acc / len(dataloader.dataset)

print(f"Training with SophiaG.")
for epoch in range(EPOCHS):
    train_loss, train_acc = train_model_with_sophia(model_for_adam_mini, train_loader, optimizer_sophia, criterion, k=10)
    test_loss, test_acc = evaluate_model(model_for_adam_mini, test_loader, criterion)
    print(f"Epoch {epoch+1}/{EPOCHS}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")