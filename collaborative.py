import pandas as pd
import numpy as np
from surprise import Reader, Dataset, SVD
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import warnings
import sys

# Ignore warning messages
warnings.filterwarnings('ignore')

# Domain Definitions (Reversed: Amazon is Source/Train, Goodreads is Target/Test)
D2_TRAIN_FILE = 'df_amazon_final.csv'    # D2 is Source (Amazon)
D1_TEST_FILE = 'df_goodreads_final.csv' # D1 is Target (Goodreads)

# SVD Model Parameters
LATENT_FACTORS = 50
SVD_EPOCHS = 30
RANDOM_STATE = 42

MIN_COMMON_BOOKS_FOR_USER_ESTIMATION = 15
USER_DATA_TEST_SIZE = 0.3


ELASTICNET_ALPHA = 1.0
L1_RATIO = 0.5


try:
    df_d2 = pd.read_csv(D2_TRAIN_FILE)
except FileNotFoundError:
    sys.exit()

# Load D2 (Amazon) data into Surprise Dataset
reader = Reader(rating_scale=(1, 10))
data_d2 = Dataset.load_from_df(df_d2[['User-ID', 'ISBN', 'Book-Rating']], reader)

trainset_d2 = data_d2.build_full_trainset()
global_mean_d2 = trainset_d2.global_mean

print(f"--- Starting SVD model training (k={LATENT_FACTORS}) ---")
model_d2 = SVD(
    n_factors=LATENT_FACTORS,
    n_epochs=SVD_EPOCHS,
    random_state=RANDOM_STATE,
    verbose=False
)
# Fit SVD model on the D2 (Amazon) data
model_d2.fit(trainset_d2)
print("SVD model training completed.")

# Extract item features (qi, bi) from the trained SVD model
d2_item_map = {}
for inner_id in range(trainset_d2.n_items):
    raw_asin = trainset_d2.to_raw_iid(inner_id)
    # qi: Item Latent Vector (Feature X)
    item_vector = model_d2.qi[inner_id]
    # bi: Item Bias (used to adjust Target Y)
    item_bias = model_d2.bi[inner_id]
    d2_item_map[raw_asin] = (item_vector, item_bias) # Map ISBN to (vector, bias) pair

# Load D1 (Goodreads) test data
try:
    df_d1 = pd.read_csv(D1_TEST_FILE)
except FileNotFoundError:
    sys.exit()

# Get unique user IDs from D1 (Target) domain
d1_user_ids = df_d1['User-ID'].unique()
print(f" Minimum common books required for estimation: {MIN_COMMON_BOOKS_FOR_USER_ESTIMATION})")

all_user_rmses = []
tested_user_count = 0

# Start Per-User Estimation Loop
for user_id in tqdm(d1_user_ids, desc="Testing D1 users"):

    # Get all ratings for the current user in D1
    user_d1_ratings_df = df_d1[df_d1['User-ID'] == user_id]

    common_books_data = []
    # Collect common books (present in both D1 user's history and D2 item map)
    for _, row in user_d1_ratings_df.iterrows():
        isbn = row['ISBN']
        rating = row['Book-Rating']

        if isbn in d2_item_map:
            # Get the SVD features (vector, bias) from D2 for the common book
            item_vector, item_bias = d2_item_map[isbn]
            common_books_data.append({
                "isbn": isbn,
                "rating": rating,
                "vector": item_vector, # qi vector (Feature X)
                "bias": item_bias      # bi bias (for adjusting Target Y)
            })

    # Skip user if they have too few common books
    if len(common_books_data) < MIN_COMMON_BOOKS_FOR_USER_ESTIMATION:
        continue

    tested_user_count += 1
    df_common = pd.DataFrame(common_books_data)

    # Split the user's data into Train (70%) for learning and Test (30%) for validation
    try:
        train_data, test_data = train_test_split(
            df_common,
            test_size=USER_DATA_TEST_SIZE,
            random_state=RANDOM_STATE
        )
    except ValueError:
        # Skips if train_test_split fails (e.g., if only one data point remains after filtering)
        continue

    # --- ElasticNet Learning (Estimating User Profile) ---
    
    # X_train: Item vectors (qi) from the training set
    X_train_vectors = np.array(list(train_data['vector']))

    # Y_train Target: Adjusted rating (Rating - bi - global_mean) 
    # This leaves only the target for the dot product + user bias (pu dot qi + bu)
    y_train_target = train_data['rating'] - train_data['bias'] - global_mean_d2

    # Initialize and train ElasticNet model (W=pu, b=bu)
    lr_model = ElasticNet(
        alpha=ELASTICNET_ALPHA,
        l1_ratio=L1_RATIO,
        fit_intercept=True, # Critical: Intercept (b) = User Bias (bu)
        random_state=RANDOM_STATE
    )
    lr_model.fit(X_train_vectors, y_train_target)

    # Extract the estimated User Profile from the model's coefficients
    p_u_estimated = lr_model.coef_      # Estimated User Factor Vector (pu)
    b_u_estimated = lr_model.intercept_ # Estimated User Bias (bu)

    # --- Performance Validation (Testing User Profile) ---

    X_test_vectors = np.array(list(test_data['vector']))
    b_i_test = test_data['bias']
    y_test_actual = test_data['rating']

    # 1. Calculate Latent Interaction: pu dot qi
    dot_product = np.dot(X_test_vectors, p_u_estimated)

    # 2. Predict Rating: mu + bu + bi + (pu dot qi)
    y_test_predicted = global_mean_d2 + b_u_estimated + b_i_test + dot_product

    # Calculate RMSE for the current user
    try:
        user_rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_predicted))
        if not np.isnan(user_rmse):
            all_user_rmses.append(user_rmse)
    except ValueError:
        continue

print("D1 user testing completed.")

print("\n--- Results ---")

if not all_user_rmses:
    print("No users met the minimum common book requirement for testing.")
else:
    final_avg_rmse = np.mean(all_user_rmses)
    print(f"Total D1 users:         {len(d1_user_ids)} users")
    print(f"Tested D1 users:        {tested_user_count} users")
    print(f"Average RMSE: {final_avg_rmse:.4f}")