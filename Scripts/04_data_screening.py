import pandas as pd


# =====================================================
# DATA SCREENING
# =====================================================

print("====================================")
print("Official Data Screening")
print("====================================")


# =====================================================
# 1. LOAD DATA
# =====================================================

data = pd.read_excel("Data/Stage 1 Official.xlsx")

# Remove completely empty rows
data = data.dropna(subset=["Participant_ID"])

print("\nOriginal dataset:")
print(f"Participants: {data['Participant_ID'].nunique()}")
print(f"Trials/rows: {len(data)}")


# =====================================================
# 2. EXCLUDE PRIOR PARTICIPANTS
# =====================================================

excluded_participants = [
    "UTS2026_t34a6rx6bgavv8d",
    "UTS2026_5r4ev8o6qdsdbcf",
    "UTS2026_uipizk4a26d1rvd",
    "UTS2026_68dmyn1ov961kvc",
    "UTS2026_2y6jytahpunvylw",
    "UTS2026_of5qslb7pq92rl5",
    "UTS2026_mnxahxn7xc8r6hc",
    "UTS2026_92cdiykrwmtmfs3"
]

before = len(data)

data = data[
    ~data["Participant_ID"].isin(excluded_participants)
]

after = len(data)

print("\nPrior participation exclusions:")
print(f"Trials removed: {before - after}")
print(f"Participants remaining: {data['Participant_ID'].nunique()}")
print(f"Trials remaining: {len(data)}")


# =====================================================
# 3. IDENTIFY COMPLETED TRIALS
# =====================================================

print("\nResponse values found:")
print(data["Response_Quadrant"].value_counts(dropna=False))


# =====================================================
# IDENTIFY INCOMPLETE / NON-RESPONSE TRIALS
# =====================================================

# Convert responses to strings and remove surrounding spaces
response = data["Response_Quadrant"].astype(str).str.strip()

# Valid responses are the four response keys
valid_responses = ["p", "q", "z", "m"]

# Identify incomplete/non-response trials
incomplete_trials = ~response.isin(valid_responses)

print("\nIncomplete/non-response trials:")
print(incomplete_trials.sum())

# Keep only trials with a valid recorded response
data = data[
    response.isin(valid_responses)
].copy()

print("\nAfter incomplete trial removal:")
print(f"Participants remaining: {data['Participant_ID'].nunique()}")
print(f"Trials remaining: {len(data)}")


# =====================================================
# 4. CHECK FOR MISSING KEY VARIABLES
# =====================================================

print("\nMissing values in key variables:")

key_variables = [
    "Participant_ID",
    "Trial_index",
    "Accuracy",
    "Response_Time_ms"
]

print(
    data[key_variables].isna().sum()
)
# =====================================================
# 5. PARTICIPANT-LEVEL SCREENING
# =====================================================

participant_summary = (
    data.groupby("Participant_ID")
    .agg(
        Completed_Trials=("Trial_index", "count"),
        Mean_Accuracy=("Accuracy", "mean"),
        Mean_RT_ms=("Response_Time_ms", "mean")
    )
    .sort_values("Completed_Trials")
)

print("\n====================================")
print("PARTICIPANT-LEVEL SUMMARY")
print("====================================")

print(participant_summary)

print("\nCompleted trials per participant:")
print(participant_summary["Completed_Trials"].describe())

print("\nAccuracy per participant:")
print(participant_summary["Mean_Accuracy"].describe())
# =====================================================
# 6. RESPONSE TIME SCREENING
# =====================================================

print("\n====================================")
print("RESPONSE TIME SUMMARY")
print("====================================")

print(data["Response_Time_ms"].describe())

print("\nLowest 10 response times:")
print(
    data[
        [
            "Participant_ID",
            "Trial_index",
            "Response_Time_ms",
            "Accuracy"
        ]
    ]
    .nsmallest(10, "Response_Time_ms")
)

print("\nHighest 10 response times:")
print(
    data[
        [
            "Participant_ID",
            "Trial_index",
            "Response_Time_ms",
            "Accuracy"
        ]
    ]
    .nlargest(10, "Response_Time_ms")
)
# =====================================================
# 7. RESPONSE TIME THRESHOLD CHECK
# =====================================================

print("\n====================================")
print("RESPONSE TIME THRESHOLD CHECK")
print("====================================")

lower_thresholds = [50, 100, 150, 200, 250, 300, 500]

print("\nTrials below lower RT thresholds:")

for threshold in lower_thresholds:
    count = (data["Response_Time_ms"] < threshold).sum()
    percentage = (count / len(data)) * 100

    print(
        f"Below {threshold} ms: "
        f"{count} trials ({percentage:.2f}%)"
    )


upper_thresholds = [5000, 10000, 15000, 30000, 60000]

print("\nTrials above upper RT thresholds:")

for threshold in upper_thresholds:
    count = (data["Response_Time_ms"] > threshold).sum()
    percentage = (count / len(data)) * 100

    print(
        f"Above {threshold} ms: "
        f"{count} trials ({percentage:.2f}%)"
    )
# =====================================================
# 8. STATISTICAL RT OUTLIER CHECK
# =====================================================

print("\n====================================")
print("STATISTICAL RT OUTLIER CHECK")
print("====================================")

# First, temporarily remove implausibly fast responses
rt_for_screening = data[
    data["Response_Time_ms"] >= 200
].copy()

# Calculate quartiles
q1 = rt_for_screening["Response_Time_ms"].quantile(0.25)
q3 = rt_for_screening["Response_Time_ms"].quantile(0.75)

iqr = q3 - q1

lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)

print(f"\nQ1: {q1:.2f} ms")
print(f"Q3: {q3:.2f} ms")
print(f"IQR: {iqr:.2f} ms")

print(f"\nLower IQR bound: {lower_bound:.2f} ms")
print(f"Upper IQR bound: {upper_bound:.2f} ms")

# Count potential upper outliers
upper_outliers = rt_for_screening[
    rt_for_screening["Response_Time_ms"] > upper_bound
]

print("\nPotential upper RT outliers:")
print(f"{len(upper_outliers)} trials")

print(
    f"Percentage: "
    f"{(len(upper_outliers) / len(rt_for_screening)) * 100:.2f}%"
)

print("\nHighest potential outliers:")
print(
    upper_outliers[
        [
            "Participant_ID",
            "Trial_index",
            "Response_Time_ms",
            "Accuracy"
        ]
    ]
    .sort_values("Response_Time_ms", ascending=False)
    .head(20)
)
# =====================================================
# 9. LOG-TRANSFORMED RT OUTLIER CHECK
# =====================================================

import numpy as np

print("\n====================================")
print("LOG-TRANSFORMED RT OUTLIER CHECK")
print("====================================")

# Use RTs of 200 ms or greater
rt_log = data[
    data["Response_Time_ms"] >= 200
].copy()

# Create natural log RT
rt_log["Log_RT"] = np.log(
    rt_log["Response_Time_ms"]
)

# Calculate IQR on log-transformed RT
log_q1 = rt_log["Log_RT"].quantile(0.25)
log_q3 = rt_log["Log_RT"].quantile(0.75)

log_iqr = log_q3 - log_q1

log_lower_bound = log_q1 - (1.5 * log_iqr)
log_upper_bound = log_q3 + (1.5 * log_iqr)

# Convert upper bound back to milliseconds
upper_rt_ms = np.exp(log_upper_bound)
lower_rt_ms = np.exp(log_lower_bound)

print(f"\nLog Q1: {log_q1:.3f}")
print(f"Log Q3: {log_q3:.3f}")
print(f"Log IQR: {log_iqr:.3f}")

print(f"\nLower RT bound: {lower_rt_ms:.2f} ms")
print(f"Upper RT bound: {upper_rt_ms:.2f} ms")

# Identify upper outliers
log_upper_outliers = rt_log[
    rt_log["Log_RT"] > log_upper_bound
]

print("\nPotential upper RT outliers:")
print(f"{len(log_upper_outliers)} trials")

print(
    f"Percentage: "
    f"{(len(log_upper_outliers) / len(rt_log)) * 100:.2f}%"
)

print("\nHighest 20 potential outliers:")

print(
    log_upper_outliers[
        [
            "Participant_ID",
            "Trial_index",
            "Response_Time_ms",
            "Accuracy"
        ]
    ]
    .sort_values(
        "Response_Time_ms",
        ascending=False
    )
    .head(20)
)
# =====================================================
# 5. SAVE SCREENED DATA
# =====================================================

data.to_csv(
    "Outputs/screened_data_step1.csv",
    index=False
)

print("\n====================================")
print("STEP 1 SCREENING COMPLETE")
print("====================================")

print("\nScreened dataset saved to:")
print("Outputs/screened_data_step1.csv")

import pandas as pd

# =====================================
# STEP 2: RT < 250 MS EXCLUSION
# =====================================

# Load the Step 1 screened dataset
screened_data = pd.read_csv(
    "Outputs/screened_data_step1.csv"
)

print("====================================")
print("STEP 2: FAST RESPONSE TIME EXCLUSION")
print("====================================")

print(f"\nTrials before RT exclusion: {len(screened_data)}")
print(
    f"Participants before RT exclusion: "
    f"{screened_data['Participant_ID'].nunique()}"
)


# -------------------------------------
# IDENTIFY TRIALS WITH RT < 250 MS
# -------------------------------------

fast_trials = screened_data[
    screened_data["Response_Time_ms"] < 250
]

print(f"\nTrials with RT < 250 ms: {len(fast_trials)}")


# -------------------------------------
# REMOVE RT < 250 MS TRIALS
# -------------------------------------

screened_data = screened_data[
    screened_data["Response_Time_ms"] >= 250
].copy()


print(f"\nTrials after RT exclusion: {len(screened_data)}")

print(
    f"Participants after RT exclusion: "
    f"{screened_data['Participant_ID'].nunique()}"
)


# =====================================
# UPDATED PARTICIPANT SUMMARY
# =====================================

participant_summary = (
    screened_data
    .groupby("Participant_ID")
    .agg(
        Completed_Trials=("Trial_index", "count"),
        Mean_Accuracy=("Accuracy", "mean"),
        Mean_RT_ms=("Response_Time_ms", "mean")
    )
)

print("\n====================================")
print("UPDATED PARTICIPANT-LEVEL SUMMARY")
print("====================================\n")

print(participant_summary)


print("\nCompleted trials per participant:")
print(participant_summary["Completed_Trials"].describe())


print("\nAccuracy per participant:")
print(participant_summary["Mean_Accuracy"].describe())


# =====================================
# SAVE STEP 2 DATASET
# =====================================

output_path = "Outputs/screened_data_step2.csv"

screened_data.to_csv(output_path, index=False)

print("\n====================================")
print("STEP 2 COMPLETE")
print("====================================")

print("\nUpdated dataset saved to:")
print(output_path)

# ====================================
# STEP 3: ABOVE-CHANCE ACCURACY TEST
# ====================================

import pandas as pd
from scipy.stats import binomtest

# Load the Step 2 screened dataset
data = pd.read_csv("Outputs/screened_data_step2.csv")

# Chance level for four possible quadrants
chance_level = 0.25

# Total number of valid trials
n_trials = len(data)

# Number of correct responses
n_correct = int(data["Accuracy"].sum())

# Overall accuracy
overall_accuracy = n_correct / n_trials

# One-sided binomial test
# Tests whether accuracy is greater than chance
result = binomtest(
    n_correct,
    n_trials,
    p=chance_level,
    alternative="greater"
)

# Print results
print("\n====================================")
print("STEP 3: ABOVE-CHANCE ACCURACY TEST")
print("====================================\n")

print(f"Valid trials: {n_trials}")
print(f"Correct responses: {n_correct}")
print(f"Overall accuracy: {overall_accuracy:.4f}")
print(f"Chance accuracy: {chance_level:.2f}")
print(f"Binomial test p-value: {result.pvalue:.10f}")

print("\n95% Confidence Interval:")
ci = result.proportion_ci(confidence_level=0.95)
print(f"Lower bound: {ci.low:.4f}")
print(f"Upper bound: {ci.high:.4f}")

if result.pvalue < 0.05:
    print("\nResult: Performance was significantly above chance.")
else:
    print("\nResult: Performance was NOT significantly above chance.")
    # ====================================
# STEP 4: PARTICIPANT-LEVEL
# ABOVE-CHANCE ACCURACY TEST
# ====================================

from scipy.stats import ttest_1samp

# Calculate each participant's mean accuracy
participant_accuracy = (
    data.groupby("Participant_ID")["Accuracy"]
    .mean()
)

# Run one-sample t-test against chance (25%)
t_statistic, two_sided_p = ttest_1samp(
    participant_accuracy,
    popmean=0.25
)

# Convert to one-sided p-value
if t_statistic > 0:
    one_sided_p = two_sided_p / 2
else:
    one_sided_p = 1 - (two_sided_p / 2)

# Print results
# ====================================
# STEP 4: PARTICIPANT-LEVEL
# ABOVE-CHANCE ACCURACY TEST
# ====================================

from scipy.stats import ttest_1samp

# Calculate each participant's mean accuracy
participant_accuracy = (
    data.groupby("Participant_ID")["Accuracy"]
    .mean()
)

# Run one-sample t-test against chance (25%)
t_statistic, two_sided_p = ttest_1samp(
    participant_accuracy,
    popmean=0.25
)

# Convert to one-sided p-value
if t_statistic > 0:
    one_sided_p = two_sided_p / 2
else:
    one_sided_p = 1 - (two_sided_p / 2)

# Print results
print("\n====================================")
print("STEP 4: PARTICIPANT-LEVEL ABOVE-CHANCE TEST")
print("====================================\n")

print(f"Number of participants: {len(participant_accuracy)}")
print(f"Mean participant accuracy: {participant_accuracy.mean():.4f}")
print(f"Standard deviation: {participant_accuracy.std():.4f}")
print(f"Chance accuracy: {chance_level:.2f}")

print(f"\nt({len(participant_accuracy) - 1}) = {t_statistic:.4f}")
print(f"One-sided p-value: {one_sided_p:.6f}")

if one_sided_p < 0.05:
    print("\nResult: Participants performed significantly above chance.")
else:
    print("\nResult: Participants did NOT perform significantly above chance.")

# Display individual participant accuracy summary
print("\nParticipant accuracy summary:")
print(participant_accuracy.describe())
# ====================================
# STEP 5: PARTICIPANT COMPLETION CHECK
# ====================================

# Count completed trials for each participant
participant_trials = (
    data.groupby("Participant_ID")
    .size()
)

print("\n====================================")
print("STEP 5: PARTICIPANT COMPLETION CHECK")
print("====================================\n")

print("Total participants:", len(participant_trials))
print("Maximum completed trials:", participant_trials.max())

print("\nParticipants meeting completion thresholds:")

max_trials = participant_trials.max()

for percentage in [0.25, 0.50, 0.75, 1.00]:
    threshold = max_trials * percentage
    n_participants = (participant_trials >= threshold).sum()

    print(
        f"{int(percentage * 100)}% completion "
        f"({threshold:.0f}+ trials): "
        f"{n_participants} participants"
    )

print("\nCompleted trial distribution:")
print(participant_trials.describe())

print("\nIndividual participant trial counts:")
print(participant_trials.sort_values())
# ====================================
# STEP 6: COMPLETION THRESHOLD
# SENSITIVITY ANALYSIS
# ====================================

from scipy.stats import ttest_1samp

print("\n====================================")
print("STEP 6: COMPLETION THRESHOLD SENSITIVITY")
print("====================================\n")

# Define completion thresholds
thresholds = {
    "All participants": 1,
    "25% completion": 63,
    "50% completion": 126,
    "75% completion": 188
}

results_list = []

for label, min_trials in thresholds.items():

    # Identify participants meeting threshold
    eligible_participants = participant_trials[
        participant_trials >= min_trials
    ].index

    # Get trial data for eligible participants
    subset = data[
        data["Participant_ID"].isin(eligible_participants)
    ]

    # Calculate participant-level accuracy
    subset_accuracy = (
        subset.groupby("Participant_ID")["Accuracy"]
        .mean()
    )

    # One-sample t-test against 25% chance
    t_stat, p_two_sided = ttest_1samp(
        subset_accuracy,
        popmean=0.25
    )

    # Convert to one-sided p-value
    if t_stat > 0:
        p_one_sided = p_two_sided / 2
    else:
        p_one_sided = 1 - (p_two_sided / 2)

    results_list.append({
        "Threshold": label,
        "Minimum Trials": min_trials,
        "Participants": len(subset_accuracy),
        "Mean Accuracy": subset_accuracy.mean(),
        "t": t_stat,
        "df": len(subset_accuracy) - 1,
        "One-sided p": p_one_sided
    })

# Convert results to table
sensitivity_results = pd.DataFrame(results_list)

print(sensitivity_results.to_string(index=False))
# ====================================
# STEP 7: PARTICIPANT COMPLETION AND PERFORMANCE
# ====================================

import pandas as pd
from scipy.stats import pearsonr, spearmanr

print("\n====================================")
print("STEP 7: PARTICIPANT COMPLETION AND PERFORMANCE")
print("====================================")

# Create participant-level summary
participant_summary = (
    data.groupby("Participant_ID")
    .agg(
        Completed_Trials=("Accuracy", "count"),
        Mean_Accuracy=("Accuracy", "mean"),
        Mean_RT_ms=("Response_Time_ms", "mean")
    )
    .reset_index()
)

print("\nParticipant-level data:")
print(participant_summary.to_string(index=False))


# ------------------------------------
# 1. COMPLETION vs ACCURACY
# ------------------------------------

pearson_acc, pearson_acc_p = pearsonr(
    participant_summary["Completed_Trials"],
    participant_summary["Mean_Accuracy"]
)

spearman_acc, spearman_acc_p = spearmanr(
    participant_summary["Completed_Trials"],
    participant_summary["Mean_Accuracy"]
)

print("\n------------------------------------")
print("COMPLETION vs ACCURACY")
print("------------------------------------")

print(f"\nPearson correlation:")
print(f"r = {pearson_acc:.4f}")
print(f"p = {pearson_acc_p:.6f}")

print(f"\nSpearman correlation:")
print(f"rho = {spearman_acc:.4f}")
print(f"p = {spearman_acc_p:.6f}")


# ------------------------------------
# 2. COMPLETION vs RESPONSE TIME
# ------------------------------------

pearson_rt, pearson_rt_p = pearsonr(
    participant_summary["Completed_Trials"],
    participant_summary["Mean_RT_ms"]
)

spearman_rt, spearman_rt_p = spearmanr(
    participant_summary["Completed_Trials"],
    participant_summary["Mean_RT_ms"]
)

print("\n------------------------------------")
print("COMPLETION vs RESPONSE TIME")
print("------------------------------------")

print(f"\nPearson correlation:")
print(f"r = {pearson_rt:.4f}")
print(f"p = {pearson_rt_p:.6f}")

print(f"\nSpearman correlation:")
print(f"rho = {spearman_rt:.4f}")
print(f"p = {spearman_rt_p:.6f}")


# ------------------------------------
# 3. SAVE PARTICIPANT SUMMARY
# ------------------------------------

participant_summary.to_csv(
    "Outputs/participant_completion_summary.csv",
    index=False
)

print("\n====================================")
print("STEP 7 COMPLETE")
print("====================================")

print("\nParticipant summary saved to:")
print("Outputs/participant_completion_summary.csv")
# ====================================
# STEP 8: COMPLETION vs RESPONSE TIME
# ====================================

import matplotlib.pyplot as plt

print("\n====================================")
print("STEP 8: COMPLETION vs RESPONSE TIME")
print("====================================")

# Create scatterplot
plt.figure(figsize=(10, 6))

plt.scatter(
    participant_summary["Completed_Trials"],
    participant_summary["Mean_RT_ms"]
)

# Add participant labels
for _, row in participant_summary.iterrows():
    plt.annotate(
        row["Participant_ID"].replace("UTS2026_", ""),
        (
            row["Completed_Trials"],
            row["Mean_RT_ms"]
        ),
        fontsize=7,
        alpha=0.7
    )

plt.xlabel("Completed Trials")
plt.ylabel("Mean Response Time (ms)")
plt.title("Relationship Between Task Completion and Mean Response Time")

plt.grid(True)

plt.tight_layout()

# Save figure
plt.savefig(
    "Outputs/completion_vs_response_time.png",
    dpi=300
)

plt.show()

print("\nScatterplot saved to:")
print("Outputs/completion_vs_response_time.png")

print("\n====================================")
print("STEP 8 COMPLETE")
print("====================================")
# ====================================
# STEP 9: LOW-COMPLETION RT DIAGNOSTICS
# ====================================

print("\n====================================")
print("STEP 9: LOW-COMPLETION RT DIAGNOSTICS")
print("====================================")

# Define low-completion participants
low_completion_ids = participant_summary.loc[
    participant_summary["Completed_Trials"] < 100,
    "Participant_ID"
]

# Get trial-level data for these participants
low_completion_data = data[
    data["Participant_ID"].isin(low_completion_ids)
].copy()

# Create RT diagnostic summary
rt_diagnostics = (
    low_completion_data
    .groupby("Participant_ID")
    .agg(
        Completed_Trials=("Response_Time_ms", "count"),
        Mean_RT_ms=("Response_Time_ms", "mean"),
        Median_RT_ms=("Response_Time_ms", "median"),
        Min_RT_ms=("Response_Time_ms", "min"),
        Max_RT_ms=("Response_Time_ms", "max"),
        RT_Over_5000ms=("Response_Time_ms",
                        lambda x: (x > 5000).sum()),
        Percent_Over_5000ms=("Response_Time_ms",
                             lambda x: (x > 5000).mean() * 100)
    )
    .sort_values("Completed_Trials")
)

print("\nResponse time diagnostics for participants")
print("with fewer than 100 completed trials:\n")

print(rt_diagnostics.to_string())


# ------------------------------------
# SAVE DIAGNOSTIC SUMMARY
# ------------------------------------

rt_diagnostics.to_csv(
    "Outputs/low_completion_rt_diagnostics.csv"
)

print("\n====================================")
print("STEP 9 COMPLETE")
print("====================================")

print("\nDiagnostic summary saved to:")
print("Outputs/low_completion_rt_diagnostics.csv")
# ====================================
# STEP 10: PARTICIPANT COMPLETION EXCLUSION
# ====================================

import pandas as pd
import os

# Load the dataset after trial-level exclusions
df = pd.read_csv("Outputs/screened_data_step2.csv")

print("====================================")
print("STEP 10: PARTICIPANT COMPLETION EXCLUSION")
print("====================================")

# Determine the maximum number of trials completed
max_trials = df.groupby("Participant_ID").size().max()

# Set minimum completion threshold
completion_threshold = 0.25

# Calculate minimum number of completed trials required
min_trials_required = int(max_trials * completion_threshold)

print()
print(f"Maximum trials: {max_trials}")
print(f"Minimum completion threshold: {completion_threshold:.0%}")
print(f"Minimum completed trials required: {min_trials_required}")

# Count completed trials for each participant
participant_trials = (
    df.groupby("Participant_ID")
    .size()
    .reset_index(name="Completed_Trials")
)

# Identify participants below the threshold
excluded_participants = participant_trials[
    participant_trials["Completed_Trials"] < min_trials_required
]

included_participants = participant_trials[
    participant_trials["Completed_Trials"] >= min_trials_required
]

print()
print("------------------------------------")
print("PARTICIPANTS EXCLUDED")
print("------------------------------------")

print(excluded_participants.to_string(index=False))

print()
print(f"Participants excluded: {len(excluded_participants)}")
print(f"Participants remaining: {len(included_participants)}")

# Remove excluded participants from the dataset
df_step10 = df[
    df["Participant_ID"].isin(included_participants["Participant_ID"])
].copy()

print()
print("------------------------------------")
print("UPDATED DATASET")
print("------------------------------------")

print(f"Trials remaining: {len(df_step10)}")
print(f"Participants remaining: {df_step10['Participant_ID'].nunique()}")

# Save the updated dataset
output_path = "Outputs/screened_data_step10.csv"

df_step10.to_csv(output_path, index=False)

print()
print("====================================")
print("STEP 10 COMPLETE")
print("====================================")
print()
print(f"Updated dataset saved to:")
print(output_path)
# ====================================
# STEP 11: POST-EXCLUSION PERFORMANCE
# ====================================

import pandas as pd
from scipy.stats import binomtest, ttest_1samp

print("\n====================================")
print("STEP 11: POST-EXCLUSION PERFORMANCE")
print("====================================\n")

# Load the dataset after participant exclusions
df_step11 = pd.read_csv(
    "Outputs/screened_data_step10.csv"
)

# ------------------------------------
# BASIC DATASET INFORMATION
# ------------------------------------

n_trials = len(df_step11)
n_participants = df_step11["Participant_ID"].nunique()

correct_responses = df_step11["Accuracy"].sum()
overall_accuracy = df_step11["Accuracy"].mean()

print(f"Participants: {n_participants}")
print(f"Valid trials: {n_trials}")
print(f"Correct responses: {int(correct_responses)}")
print(f"Overall accuracy: {overall_accuracy:.4f}")

# ------------------------------------
# OVERALL BINOMIAL TEST
# ------------------------------------

chance_accuracy = 0.25

binomial_result = binomtest(
    int(correct_responses),
    n_trials,
    p=chance_accuracy,
    alternative="greater"
)

print("\n------------------------------------")
print("OVERALL ABOVE-CHANCE TEST")
print("------------------------------------\n")

print(f"Chance accuracy: {chance_accuracy}")
print(f"Binomial test p-value: {binomial_result.pvalue:.6f}")

if binomial_result.pvalue < 0.05:
    print("\nResult: Overall performance was significantly above chance.")
else:
    print("\nResult: Overall performance was NOT significantly above chance.")

# ------------------------------------
# PARTICIPANT-LEVEL ACCURACY
# ------------------------------------

participant_accuracy = (
    df_step11
    .groupby("Participant_ID")["Accuracy"]
    .mean()
)

mean_participant_accuracy = participant_accuracy.mean()
sd_participant_accuracy = participant_accuracy.std()

print("\n------------------------------------")
print("PARTICIPANT-LEVEL ACCURACY")
print("------------------------------------\n")

print(f"Mean participant accuracy: {mean_participant_accuracy:.4f}")
print(f"Standard deviation: {sd_participant_accuracy:.4f}")

# ------------------------------------
# PARTICIPANT-LEVEL ABOVE-CHANCE TEST
# ------------------------------------

t_stat, two_sided_p = ttest_1samp(
    participant_accuracy,
    popmean=chance_accuracy
)

# Convert to one-sided p-value
if t_stat > 0:
    one_sided_p = two_sided_p / 2
else:
    one_sided_p = 1 - (two_sided_p / 2)

df_degrees = len(participant_accuracy) - 1

print("\n------------------------------------")
print("PARTICIPANT-LEVEL ABOVE-CHANCE TEST")
print("------------------------------------\n")

print(f"t({df_degrees}) = {t_stat:.4f}")
print(f"One-sided p-value: {one_sided_p:.6f}")

if one_sided_p < 0.05:
    print("\nResult: Participants performed significantly above chance.")
else:
    print("\nResult: Participants did NOT perform significantly above chance.")

# ------------------------------------
# RESPONSE TIME SUMMARY
# ------------------------------------

rt = df_step11["Response_Time_ms"]

print("\n------------------------------------")
print("RESPONSE TIME SUMMARY")
print("------------------------------------\n")

print(rt.describe())

print("\n====================================")
print("STEP 11 COMPLETE")
print("====================================")
print(df_step11.columns.tolist())
# ====================================
# STEP 12: INSPECT TEXTURE PAIRS
# ====================================

# ====================================
# STEP 12: TEXTURE SUMMARY
# ====================================

print("\n====================================")
print("STEP 12: TEXTURE SUMMARY")
print("====================================\n")

df_step12 = pd.read_csv(
    "Outputs/screened_data_step10.csv"
)

# Get all textures appearing anywhere
all_textures = sorted(
    set(df_step12["Majority_Texture"].dropna())
    | set(df_step12["Minority_Texture"].dropna())
)

print("ALL UNIQUE TEXTURES:")
print(all_textures)

print(f"\nTotal number of unique textures: {len(all_textures)}")

# Count unique pairs
texture_pairs = (
    df_step12[
        ["Majority_Texture", "Minority_Texture"]
    ]
    .drop_duplicates()
)

print(f"\nTotal number of unique presented pairs: {len(texture_pairs)}")

print("\nFIRST 20 UNIQUE PAIRS:")
print(texture_pairs.head(20).to_string(index=False))

print("\n====================================")
print("STEP 12 COMPLETE")
print("====================================")