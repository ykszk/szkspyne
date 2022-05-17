import pandas as pd
import matplotlib.pyplot as plt


def plot_history(df, set_ylim=True):
    df = pd.DataFrame((df['train_loss'].dropna().reset_index(drop=True),
                       df['val_loss'].dropna().reset_index(drop=True))).T
    df.plot(y=['train_loss', 'val_loss'])
    if set_ylim and len(df) > 1:
        plt.ylim(df[['train_loss', 'val_loss']].min().min(),
                 sorted(df['train_loss'])[-2])
