import matplotlib.pyplot as plt


def plot_training_history(
    model_history,
    output_path=None,
    show=True,
):
    history = model_history.history

    train_loss = history["loss"]
    val_loss = history["val_loss"]

    mae_key = _first_existing_key(
        history,
        ["mean_absolute_error", "mae"],
    )
    val_mae_key = _first_existing_key(
        history,
        ["val_mean_absolute_error", "val_mae"],
    )

    if mae_key is None or val_mae_key is None:
        raise KeyError(
            "Cannot plot MAE history. Expected mean_absolute_error/mae and "
            "val_mean_absolute_error/val_mae in model_history.history."
        )

    train_mae = history[mae_key]
    val_mae = history[val_mae_key]

    plt.figure(figsize=(20, 8))

    plt.subplot(121)
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title("Model Loss per Epoch (MSE)")
    plt.ylabel("Loss (MSE)")
    plt.xlabel("Epoch")
    plt.legend(["Training", "Validation"], loc="best")

    plt.subplot(122)
    plt.plot(train_mae)
    plt.plot(val_mae)
    plt.title("Mean Absolute Error per Epoch (MAE)")
    plt.ylabel("Mean Absolute Error (MAE)")
    plt.xlabel("Epoch")
    plt.legend(["Training", "Validation"], loc="best")

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def _first_existing_key(history, keys):
    for key in keys:
        if key in history:
            return key

    return None
