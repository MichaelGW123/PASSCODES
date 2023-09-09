import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Read the CSV file into a pandas DataFrame
csv_file = "/home/morpheus/Research/DeepLearningEntropy/Source_Files/entropy.csv"
data = pd.read_csv(csv_file, header=None)

sample_size = 130000000
# Use a random_state for reproducibility
sampled_data = data.sample(n=sample_size, random_state=42)

# Step 3: Create a KDE plot using seaborn based on the sampled data
sns.set(style="whitegrid")  # Optional: Set the plot style

# Adjust the bandwidth (bw_method) parameter as needed to control the smoothness of the KDE plot
# Use the column by positional index (0 in this case since it's the first column)
sns.kdeplot(data=sampled_data.iloc[:, 0], bw_method=0.5)

# Step 3: Customize the plot (optional)
plt.title("Kernel Density Estimation (KDE) Plot")
plt.xlabel("X-axis Label")  # Replace with your desired X-axis label
plt.ylabel("Density")  # Replace with your desired Y-axis label

# Step 4: Show the KDE plot
plt.show()
