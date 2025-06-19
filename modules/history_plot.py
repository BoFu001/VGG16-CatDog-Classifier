import matplotlib.pyplot as plt

def plot_history(history, title="Training History"):
    if history is None:
        print("No training history available to plot.")
        return

    # Ensure history is in dictionary format
    history_dict = history if isinstance(history, dict) else history.history

    plt.figure(figsize=(8, 5))
    plt.plot(history_dict.get("accuracy", []), label="Train Acc")
    plt.plot(history_dict.get("val_accuracy", []), label="Val Acc")
    plt.plot(history_dict.get("loss", []), label="Train Loss", linestyle="--")
    plt.plot(history_dict.get("val_loss", []), label="Val Loss", linestyle="--")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()