import itertools

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

sns.set()
sns.set_theme(style="white")
FONTSIZE = 16


def load_markdown_table(pth: str) -> pd.DataFrame:
    df = pd.read_csv(
        filepath_or_buffer=pth, sep="|", skipinitialspace=True, skiprows=[1]
    )
    df.columns = [x.strip() for x in df.columns]
    return df.dropna(axis=1, how="all")


def postprocess_column(key: str, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Given key and df, return (x, y) series for plotting
    """
    loss = pd.to_numeric(df[key], errors="coerce")
    loss = loss.dropna()
    return df["Epoch"].iloc[loss.index], loss


# load loss plots
foodbb_df = load_markdown_table("foodbb_df.csv")
x251_df = load_markdown_table("x251_df.csv")

print(foodbb_df)


def plot_df(
    df: pd.DataFrame,
    val_key: str,
    ds_name: str,
    ax_train: Axes,
    ax_val: Axes,
    color_train: str,
    color_val: str,
) -> tuple[Axes, Axes]:
    # Make a lineplot of (training loss, validation loss)
    # training
    train_epochs, train_loss = postprocess_column("Training Loss", df)
    ax_train.plot(
        train_epochs, train_loss, label=f"{ds_name} Training", color=color_train
    )
    ax_train.set_xlabel("Epochs", fontsize=FONTSIZE)
    ax_train.set_ylabel("Loss", fontsize=FONTSIZE)

    # validation
    val_epochs, val_loss = postprocess_column(val_key, df)
    ax_val.plot(val_epochs, val_loss, label=f"{ds_name} Validation", color=color_val)
    ax_val.set_xlabel("Epochs", fontsize=FONTSIZE)
    ax_val.set_ylabel("Accuracy", fontsize=FONTSIZE)
    return ax_train, ax_val


#
fig, ax_train = plt.subplots(dpi=300)
ax_val = ax_train.twinx()
plot_df(
    x251_df, "foodx251-val_cosine_ap", "FoodX-251", ax_train, ax_val, "red", "orange"
)
plot_df(
    foodbb_df, "foodbb-val_cosine_ap", "Combined", ax_train, ax_val, "blue", "purple"
)

ax_train.set_title("Finetuning Loss of jina-clip-v2", fontsize=FONTSIZE)
axes = [ax_train, ax_val]

train_lines, train_labels = ax_train.get_legend_handles_labels()
val_lines, val_labels = ax_val.get_legend_handles_labels()
lines = [train_lines[0], val_lines[0], train_lines[1], val_lines[1]]
labels = [train_labels[0], val_labels[0], train_labels[1], val_labels[1]]
ax_train.legend(lines, labels, loc="center right")
plt.tight_layout()
ax_train.tick_params(axis="x", labelsize=14)
ax_train.tick_params(axis="y", labelsize=14)
ax_val.tick_params(axis="y", labelsize=14)
plt.savefig("loss.png")
