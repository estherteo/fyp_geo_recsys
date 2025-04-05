import os.path
import pickle

import numpy as np
import pandas as pd
from pytorch_widedeep import Trainer
from pytorch_widedeep.callbacks import EarlyStopping
from pytorch_widedeep.preprocessing import WidePreprocessor, TabPreprocessor
from pytorch_widedeep.models import Wide, TabMlp, WideDeep
from pytorch_widedeep.metrics import  F1Score, Precision, Recall

BATCH_SIZE=32768
NUM_EPOCHS=50
REPROCESS = False

train = pd.read_parquet('/workspace/dataset_challenge/train.parquet')
val = pd.read_parquet('/workspace/dataset_challenge/val.parquet')
test = pd.read_parquet('/workspace/dataset_challenge/test.parquet')
test = test[test['business_is_open'] == 1]

train['review_stars'] *= 5
train = train.astype({'review_stars': int})

val['review_stars'] *= 5
val = val.astype({'review_stars': int})

test['review_stars'] *= 5
test = test.astype({'review_stars': int})

df_columns = ['business_id',
             'review_stars',
             'user_id',
             'review_dist_to_centroid',
             'review_bbox_area',
             'review_min_bbox_area',
             'review_area_ratio',
             'user_review_count',
             'user_useful',
             'user_funny',
             'user_cool',
             'user_fans',
             'user_average_stars',
             'user_compliment_score',
             'user_elite_count',
             'user_lifetime',
             'user_num_friends',
             'user_num_tips',
             'business_stars',
             'business_review_count',
             'business_is_open',
             'business_BikeParking',
             'business_WheelchairAccessible',
             'business_total_hours',
             'business_Fri',
             'business_Mon',
             'business_Sat',
             'business_Sun',
             'business_Thu',
             'business_Tue',
             'business_Wed',
             'business_checkins',
             'business_num_years_user_visit_business',
             'review_visited',
             'user_isExplorer']

# Define the 'column set up'
wide_cols = [ # Capture “memorization” by modeling feature interactions directly with linear relationships.
            'business_id',
            #'review_stars',
            'user_id',
            'business_stars',
            'business_review_count',
            'business_BikeParking',
            'business_WheelchairAccessible',
            'business_total_hours',
            'business_Fri',
            'business_Mon',
            'business_Sat',
            'business_Sun',
            'business_Thu',
            'business_Tue',
            'business_Wed',
            'business_checkins',
            'business_num_years_user_visit_business',
            'user_review_count',
            'user_useful',
            'user_funny',
            'user_cool',
            'user_fans',
            'user_average_stars',
            'user_compliment_score',
            'user_elite_count',
            'user_lifetime',
            'user_num_friends',
            'user_num_tips',
          ]

crossed_cols = [('user_average_stars', 'business_stars'),
                ('business_checkins', 'user_num_tips'),
                ('user_review_count', 'business_review_count'),
                ('user_average_stars', 'business_review_count'),
                ('user_elite_count', 'business_stars'),
                ('business_num_years_user_visit_business', 'user_lifetime')
                ]

cat_embed_cols = [
    'user_id',
    'business_id',
    'business_BikeParking',
    'business_WheelchairAccessible',
]

continuous_cols = [
    #'review_stars',
    'business_stars',
    'business_review_count',
    'business_total_hours',
    'business_Fri',
    'business_Mon',
    'business_Sat',
    'business_Sun',
    'business_Thu',
    'business_Tue',
    'business_Wed',
    'business_checkins',
    'business_num_years_user_visit_business',
    'user_review_count',
    'user_useful',
    'user_funny',
    'user_cool',
    'user_fans',
    'user_average_stars',
    'user_compliment_score',
    'user_elite_count',
    'user_lifetime',
    'user_num_friends',
    'user_num_tips',
]

target = "review_stars"
target_train = train[target].values
target_val = val[target].values
target_test = test[target].values

wide_preprocessor = WidePreprocessor(wide_cols=wide_cols, crossed_cols=crossed_cols)
if not os.path.isfile("./dataset_challenge/wide_preprocessed") or REPROCESS:
    # prepare the data
    print("Pre-processing for wide")
    X_wide_train = wide_preprocessor.fit_transform(train)
    X_wide_val = wide_preprocessor.transform(val)
    X_wide_test = wide_preprocessor.transform(test)

    with open("./dataset_challenge/wide_preprocessed", "wb") as f:
        pickle.dump([X_wide_train, X_wide_val, X_wide_test, wide_preprocessor], f, protocol=4)
else:
    print("Loading pre-processed wide")
    with open("./dataset_challenge/wide_preprocessed", "rb") as f:
        X_wide_train, X_wide_val, X_wide_test, wide_preprocessor = pickle.load(f)

tab_preprocessor = TabPreprocessor(
        cat_embed_cols=cat_embed_cols, continuous_cols=continuous_cols  # type: ignore[arg-type]
)

if not os.path.isfile("./dataset_challenge/tab_preprocessed") or REPROCESS:
    print("Pre-processing for deeptabular")
    X_tab_train = tab_preprocessor.fit_transform(train)
    X_tab_val = tab_preprocessor.transform(val)
    X_tab_test = tab_preprocessor.transform(test)

    with open("./dataset_challenge/tab_preprocessed", "wb") as f:
        pickle.dump([X_tab_train, X_tab_val, X_tab_test, tab_preprocessor], f, protocol=4)
else:
    print("Loading pre-processed deeptabular")
    with open("./dataset_challenge/tab_preprocessed", "rb") as f:
        X_tab_train, X_tab_val, X_tab_test, tab_preprocessor = pickle.load(f)

# build the model
print("Building the model")
wide = Wide(input_dim=np.unique(X_wide_train).shape[0], pred_dim=6)
tab_mlp = TabMlp(
    column_idx=tab_preprocessor.column_idx,
    cat_embed_input=tab_preprocessor.cat_embed_input,
    continuous_cols=tab_preprocessor.continuous_cols,
    cont_norm_layer=None,
    mlp_hidden_dims=[1024, 512, 256],
    mlp_activation="relu",
)

model = WideDeep(wide=wide, deeptabular=tab_mlp, pred_dim=6)

# train and validate
print("Training the model")

trainer = Trainer(model, objective="categorical_cross_entropy", metrics=[F1Score(), Precision(), Recall()],
                  callbacks=[EarlyStopping(patience=5, monitor="val_f1", restore_best_weights=True)])

"""
Precision_at_k(k=10, n_cols=BATCH_SIZE),
  Recall_at_k(k=10, n_cols=BATCH_SIZE),
  NDCG_at_k(k=10, n_cols=BATCH_SIZE),
  HitRatio_at_k(k=10, n_cols=BATCH_SIZE),
"""

trainer.fit(
    X_train={"X_wide": X_wide_train, "X_tab": X_tab_train, "target": target_train},
    X_val={"X_wide": X_wide_val, "X_tab": X_tab_val, "target": target_val},
    n_epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    drop_last=True,
)

print("Testing the model")
# predict on test
preds = trainer.predict(X_wide=X_wide_test, X_tab=X_tab_test)

print('preds:', preds)

# Save and load
trainer.save(path="./model_weights", save_state_dict=True)

# Save Preds
np.save("/root/preds.npy", preds)