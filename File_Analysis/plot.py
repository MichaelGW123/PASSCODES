import seaborn as sns
import matplotlib.pyplot as plt
import dask.dataframe as dd
import random

plot_type = 1

sample_size = 9000000
csv_file = "/home/morpheus/Research/DeepLearningEntropy/Source_Files/6sig_hashesorg2019.csv"

# Read the CSV file into a Dask DataFrame
df = dd.read_csv(csv_file, dtype={'1': 'object'})

# Calculate the fraction based on the sample size and total number of rows
total_rows = len(df)
fraction = sample_size / total_rows

# Randomly sample rows
sampled_df = df.sample(frac=fraction, random_state=random.seed(42))

del df

# Compute the result and retrieve the sampled data
sampled_data = sampled_df.compute()

# Step 3: Create a KDE plot using seaborn based on the sampled data
sns.set(style="whitegrid")  # Optional: Set the plot style

if (plot_type == 1):
    # Adjust the bandwidth (bw_method) parameter as needed to control the smoothness of the KDE plot
    # Use the column by positional index (1 in this case since it's the first column)
    sns.kdeplot(data=sampled_data.iloc[:, 1], bw_method=0.5)
    plt.title("Kernel Density Estimation (KDE) Plot")
elif (plot_type == 2):
    # Create a histogram using Seaborn
    # Adjust the number of bins as needed
    sns.histplot(data=sampled_data.iloc[:, 1], bins=25)
    plt.title("Histogram Plot")

# Step 3: Customize the plot (optional)
plt.xlabel("Number of Passwords")  # Replace with your desired X-axis label
plt.ylabel("Entropy")  # Replace with your desired Y-axis label

# Step 4: Show the KDE plot
plt.show()
