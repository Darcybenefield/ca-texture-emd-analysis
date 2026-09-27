import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# --------------------------------------------------
# 1. Load the final screened dataset
# --------------------------------------------------

data = pd.read_csv("Outputs/screened_data_final.csv")
emd = pd.read_csv("Outputs/emd_results_final.csv")

# --------------------------------------------------
# 2. Add EMD to the trial-level dataset
# --------------------------------------------------

data["EMD"] = emd["EMD"]

# --------------------------------------------------
# 3. Log-transform response time
# --------------------------------------------------

data["log_RT"] = np.log(data["Response_Time_ms"])

# --------------------------------------------------
# 4. Fit the linear mixed-effects model
# --------------------------------------------------

model = smf.mixedlm(
    "log_RT ~ EMD",
    data=data,
    groups=data["Participant_ID"]
)

result = model.fit()

print("\nMODEL RESULTS")
print("=" * 60)
print(result.summary())

# --------------------------------------------------
# 5. Calculate predicted values and residuals
# --------------------------------------------------

data["predicted_log_RT"] = result.fittedvalues

data["residual"] = result.resid

data["absolute_residual"] = data["residual"].abs()

# --------------------------------------------------
# 6. Display the 20 largest residuals
# --------------------------------------------------

largest_residuals = (
    data
    .sort_values("absolute_residual", ascending=False)
    .head(20)
)

columns_to_show = [
    "Participant_ID",
    "Trial_index",
    "Majority_Texture",
    "Minority_Texture",
    "EMD",
    "Accuracy",
    "Response_Time_ms",
    "log_RT",
    "predicted_log_RT",
    "residual",
    "absolute_residual"
]

print("\n20 LARGEST RESIDUALS")
print("=" * 60)

print(
    largest_residuals[columns_to_show]
    .to_string(index=False)
)

# --------------------------------------------------
# 7. Save the residual dataset
# --------------------------------------------------

data.to_csv(
    "Outputs/residual_analysis.csv",
    index=False
)

largest_residuals[columns_to_show].to_csv(
    "Outputs/largest_residuals.csv",
    index=False
)

print("\nFiles saved:")
print("Outputs/residual_analysis.csv")
print("Outputs/largest_residuals.csv")
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load the final screened dataset
data = pd.read_csv("Outputs/screened_data_final.csv")
emd = pd.read_csv("Outputs/emd_results_final.csv")

# Add EMD to the trial-level dataset
data["EMD"] = emd["EMD"]

# Log-transform response time
data["log_RT"] = np.log(data["Response_Time_ms"])

# Fit the linear mixed-effects model
model = smf.mixedlm(
    "log_RT ~ EMD",
    data=data,
    groups=data["Participant_ID"]
)

result = model.fit()

print("\nMODEL RESULTS")
print("=" * 60)
print(result.summary())

# Calculate predicted values and residuals
data["predicted_log_RT"] = result.fittedvalues
data["residual"] = result.resid
data["absolute_residual"] = data["residual"].abs()

# Find the 20 largest residuals
largest_residuals = (
    data
    .sort_values("absolute_residual", ascending=False)
    .head(20)
)

columns_to_show = [
    "Participant_ID",
    "Trial_index",
    "Majority_Texture",
    "Minority_Texture",
    "EMD",
    "Accuracy",
    "Response_Time_ms",
    "log_RT",
    "predicted_log_RT",
    "residual",
    "absolute_residual"
]

print("\n20 LARGEST RESIDUALS")
print("=" * 60)

print(
    largest_residuals[columns_to_show]
    .to_string(index=False)
)

# Save the complete residual dataset
data.to_csv(
    "Outputs/residual_analysis.csv",
    index=False
)

# Save the 20 largest residuals
largest_residuals[columns_to_show].to_csv(
    "Outputs/largest_residuals.csv",
    index=False
)

print("\nFiles saved:")
print("Outputs/residual_analysis.csv")
print("Outputs/largest_residuals.csv")

# --------------------------------------------------
# 8. Plot residuals against EMD
# --------------------------------------------------

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

plt.scatter(
    data["EMD"],
    data["residual"],
    alpha=0.25,
    s=15
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Earth Mover's Distance (EMD)")
plt.ylabel("Residual (log RT)")
plt.title("Residuals versus EMD")

plt.tight_layout()

plt.savefig(
    "Outputs/residuals_vs_emd.png",
    dpi=300
)

plt.show()