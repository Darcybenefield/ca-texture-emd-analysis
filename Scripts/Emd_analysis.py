import pandas as pd
import numpy as np
import ast


# =====================================================
# FUNCTIONS
# =====================================================

def extract_textures(matrix, location):
    """
    Splits a trial image into four quadrants and returns:
        majority_texture
        minority_texture
    """

    rows, cols = matrix.shape

    # Ensure even dimensions
    if rows % 2 != 0:
        matrix = matrix[:-1, :]
        rows -= 1

    if cols % 2 != 0:
        matrix = matrix[:, :-1]
        cols -= 1

    mid_row = rows // 2
    mid_col = cols // 2

    quadrants = {
        "UL": matrix[:mid_row, :mid_col],
        "UR": matrix[:mid_row, mid_col:],
        "LL": matrix[mid_row:, :mid_col],
        "LR": matrix[mid_row:, mid_col:]
    }

    minority = quadrants[location]

    # Use the first remaining quadrant as the representative majority
    majority = next(
        quad for name, quad in quadrants.items()
        if name != location
    )

    return majority, minority


def calculate_histogram(texture):
    """
    Calculates a NORMALISED histogram of every possible
    2x2 binary neighbourhood (16 bins).
    """

    histogram = np.zeros(16, dtype=float)

    rows, cols = texture.shape

    for r in range(rows - 1):
        for c in range(cols - 1):

            block = texture[r:r+2, c:c+2]

            index = (
                int(block[0, 0]) * 8 +
                int(block[0, 1]) * 4 +
                int(block[1, 0]) * 2 +
                int(block[1, 1])
            )

            histogram[index] += 1

    # Normalise
    histogram /= histogram.sum()

    return histogram
def build_ground_distance_matrix():
    """
    Creates a 16 × 16 Euclidean distance matrix for the
    16 possible 2×2 binary neighbourhoods.
    """

    patterns = []

    # Generate all 16 binary patterns
    for i in range(16):

        pattern = [
            (i >> 3) & 1,
            (i >> 2) & 1,
            (i >> 1) & 1,
            i & 1
        ]

        patterns.append(pattern)

    patterns = np.array(patterns)

    ground = np.zeros((16, 16))

    # Euclidean distance between every pair
    for i in range(16):
        for j in range(16):

            ground[i, j] = np.linalg.norm(
                patterns[i] - patterns[j]
            )

    return ground
ground = build_ground_distance_matrix()

import ot


def calculate_emd(histogram1, histogram2, ground_distance):
    """
    Calculates Earth Mover's Distance between
    two normalised histograms.
    """

    emd = ot.emd2(
        histogram1,
        histogram2,
        ground_distance
    )

    return emd

    ground = build_ground_distance_matrix()

print("Ground matrix built.")

# =====================================================
# MAIN ANALYSIS
# =====================================================

print("Loading data...")

data = pd.read_excel("Data/Stage 2.xlsx")

ground = build_ground_distance_matrix()

results = []

print(f"Processing {len(data)} trials...\n")

for i in range(len(data)):

    # Read trial matrix
    try:
        matrix = np.array(
            ast.literal_eval(
                data.loc[i, "Matrix"]
            )
        )

    except Exception as e:
        print(f"\nError on row {i}")
        print(f"Texture ID: {data.loc[i, 'Texture iD']}")
        print("Matrix value:")
        print(repr(data.loc[i, "Matrix"]))
        raise

    # Read minority location
    location = data.loc[i, "Minority Location"]

    # Extract textures
    majority_texture, minority_texture = extract_textures(
        matrix,
        location
    )

    # Generate histograms
    majority_hist = calculate_histogram(majority_texture)
    minority_hist = calculate_histogram(minority_texture)

    # Calculate EMD
    emd = calculate_emd(
        majority_hist,
        minority_hist,
        ground
    )

    # Store result
    results.append({

        "Texture iD": data.loc[i, "Texture iD"],

        "Majority Texture iD":
            data.loc[i, "Majority Texture iD"],

        "Minority Texture iD":
            data.loc[i, "Minority Texture iD"],

        "Minority Location": location,

        "EMD": emd
    })

print("Finished!")

results = pd.DataFrame(results)

results.to_csv(
    "Outputs/emd_results.csv",
    index=False
)

print()
print(results.head())
print()
print(results["EMD"].describe())

python3 -c "
import pandas as pd

df = pd.read_excel('Data/Stage 1 Official.xlsx')

print('Original participants:', df['Participant_ID'].nunique())
print()
print('ALL ORIGINAL PARTICIPANT IDs:')
for pid in sorted(df['Participant_ID'].dropna().unique()):
    print(pid)