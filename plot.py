import matplotlib.pyplot as plt
import re

# Initialize data storage
optimizers = {}
current_optimizer = None

# Read the log file
with open('log.txt', 'r') as file:
    for line in file:
        # Detect the optimizer
        if "Training with" in line:
            current_optimizer = line.split(" ")[2].strip()
            optimizers[current_optimizer] = {"epochs": [], "losses": []}
        
        # Extract epoch and training loss
        match = re.search(r"Epoch (\d+)/\d+: Train Loss: ([\d.]+)", line)
        if match and current_optimizer:
            epoch = int(match.group(1))
            train_loss = float(match.group(2))
            optimizers[current_optimizer]["epochs"].append(epoch)
            optimizers[current_optimizer]["losses"].append(train_loss)

# Plot the data
plt.figure(figsize=(10, 6))
for optimizer, data in optimizers.items():
    plt.plot(data["epochs"], data["losses"], label=optimizer)

plt.title("Training Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Training Loss")
plt.legend()
plt.grid(True)
plt.savefig(f'training_loss.png')
plt.close()
