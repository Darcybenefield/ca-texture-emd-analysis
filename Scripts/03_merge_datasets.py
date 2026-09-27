import pandas as pd

print("====================================")
print("Merge Behavioural and EMD Datasets")
print("====================================")

# Load datasets
behaviour = pd.read_excel("Data/Stage 1.xlsx")
emd = pd.read_csv("Outputs/emd_results.csv")
master = behaviour.merge(
    emd,
    left_on="Trial iD",
    right_on="Texture iD",
    how="left"
)

print("\nMerge complete!")
print(f"Rows in merged dataset: {len(master)}")

print("\nMissing EMD values:")
print(master["EMD"].isna().sum())

print("\nMerged columns:")
print(master.columns)

print("\nFirst five rows:")
print(master.head())
# Save master dataset
master.to_csv(
    "Outputs/master_dataset.csv",
    index=False
)

print("\nMaster dataset saved to Outputs/master_dataset.csv")