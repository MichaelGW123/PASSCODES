import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def create_entropy_kde_plot(csv_file):
    # Read the CSV file into a DataFrame with no header
    df = pd.read_csv(csv_file, header=None)

    # Assuming the first column contains passwords and the second column contains entropy
    entropy_column = 1

    # Create a KDE plot of the entropy values
    sns.set(style="whitegrid")
    sns.kdeplot(df.iloc[:, entropy_column], fill=True, color='skyblue')

    # Add labels and title
    plt.xlabel('Entropy')
    plt.ylabel('Density')
    plt.title('Kernel Density Estimation (KDE) of Password Entropy')

    # Show the plot
    plt.show()


if __name__ == "__main__":
    # Replace 'your_file.csv' with the actual path to your unlabeled CSV file containing password and entropy data
    csv_file = "/home/morpheus/Research/DeepLearningEntropy/Source_Files/hashesorg2019.csv"
    create_entropy_kde_plot(csv_file)
