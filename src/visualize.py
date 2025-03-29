import seaborn as sns
import matplotlib.pyplot as plt

def plot_feature_vs_target(df, target_col="Up", top_n=10):
    """目的変数と各特徴量の関係を可視化"""
    features = df.drop([target_col]).columns[:top_n]

    for col in features:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=target_col, y=col, data=df, palette="Set2")
        plt.title(f"{col} vs {target_col}")
        plt.tight_layout()
        plt.show()

def plot_violin(df, feature, target="Up"):
    plt.figure(figsize=(6, 4))
    sns.violinplot(x=target, y=feature, data=df, palette="Set2", cut=0)
    plt.title(f"{feature} vs {target} (Violin Plot)")
    plt.tight_layout()
    plt.show()
