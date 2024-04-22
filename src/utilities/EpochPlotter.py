import matplotlib.pyplot as plt

def plot_epoch_loss(filepath):
    epochs = []
    losses = []

    with open(filepath, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            epoch = int(parts[0].split()[1])
            loss = float(parts[1].split()[2])
            epochs.append(epoch)
            losses.append(loss)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, losses, label='Training Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
plot_epoch_loss('F:/PythonProjects/EmotionRecognition/data/training_loss.txt')
