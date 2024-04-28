import matplotlib.pyplot as plt

def plot_epoch_loss(filepath):
    epochs = []
    losses = []

    with open(filepath, 'r') as file:
        for line in file:
            if line.strip():  # Ensure the line is not empty
                parts = line.strip().split(',')  # Split the line at commas
                if len(parts) >= 2:  # Ensure there are at least two parts
                    epoch_part = parts[0].split()  # Split the first part on whitespace
                    loss_part = parts[1].split()  # Split the second part on whitespace
                    if len(epoch_part) >= 2 and len(loss_part) >= 2:  # Ensure each part has enough elements
                        try:
                            epoch = int(epoch_part[1])  # Convert the second element to integer
                            loss = float(loss_part[1])  # Convert the second element to float
                            epochs.append(epoch)
                            losses.append(loss)
                        except ValueError as e:
                            print(f"Error converting line to int or float: {line}. Error: {e}")
                    else:
                        print(f"Not enough data to parse in line: {line}")
                else:
                    print(f"Not enough parts after splitting by comma: {line}")
            else:
                print("Skipped empty line")

    if epochs and losses:  # Check if data was added to lists
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, losses, label='Training Loss', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("No data to plot.")

# Example usage
plot_epoch_loss('F:/PythonProjects/EmotionRecognition/data/training_loss.txt')
