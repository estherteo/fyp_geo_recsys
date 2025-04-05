import os.path
import pickle

import numpy as np
import pandas as pd
from pytorch_widedeep.callbacks import EarlyStopping
from pytorch_widedeep import Trainer
from pytorch_widedeep.preprocessing import WidePreprocessor, TabPreprocessor
from pytorch_widedeep.models import Wide, TabMlp, WideDeep
from pytorch_widedeep.metrics import F1Score, Precision, Recall

BATCH_SIZE=32768
NUM_EPOCHS=10
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
              'user_isExplorer'
              'review_visited'
              ]

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
            'review_dist_to_centroid',
            'review_bbox_area',
            'review_min_bbox_area',
            'review_area_ratio',
          ]

crossed_cols = [('user_average_stars', 'business_stars'),
                ('business_checkins', 'user_num_tips'),
                ('user_review_count', 'business_review_count'),
                ('user_average_stars', 'business_review_count'),
                ('user_elite_count', 'business_stars'),
                ('business_num_years_user_visit_business', 'user_lifetime'),
                ('review_area_ratio', 'business_checkins'),
                ('review_area_ratio', 'business_num_years_user_visit_business'),
                ('review_area_ratio', 'business_stars'),
                ('review_area_ratio', 'business_review_count'),
                ('user_isExplorer', 'business_checkins'),
                ('user_isExplorer', 'business_num_years_user_visit_business'),
                ('user_isExplorer', 'business_stars'),
                ('user_isExplorer', 'business_review_count'),
                ]

cat_embed_cols = [
    'user_id',
    'business_id',
    'business_BikeParking',
    'business_WheelchairAccessible',
    'user_isExplorer',
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
    'review_dist_to_centroid',
    'review_bbox_area',
    'review_min_bbox_area',
    'review_area_ratio',
]
target = "review_stars"
target_train = train[target].values
target_val = val[target].values
target_test = test[target].values


wide_preprocessor = WidePreprocessor(wide_cols=wide_cols, crossed_cols=crossed_cols)
if REPROCESS or not os.path.isfile("/workspace/dataset_challenge/geo_wide_preprocessed"):
    # prepare the data
    print("Pre-processing for wide")
    X_wide_train = wide_preprocessor.fit_transform(train)
    X_wide_val = wide_preprocessor.transform(val)
    X_wide_test = wide_preprocessor.transform(test)

    with open("/workspace/dataset_challenge/geo_wide_preprocessed", "wb") as f:
        pickle.dump([X_wide_train, X_wide_val, X_wide_test, wide_preprocessor], f, protocol=4)
else:
    print("Loading pre-processed wide")
    with open("/workspace/dataset_challenge/geo_wide_preprocessed", "rb") as f:
        X_wide_train, X_wide_val, X_wide_test, wide_preprocessor = pickle.load(f)

tab_preprocessor = TabPreprocessor(
        cat_embed_cols=cat_embed_cols, continuous_cols=continuous_cols  # type: ignore[arg-type]
)
if REPROCESS or not os.path.isfile("/workspace/dataset_challenge/geo_tab_preprocessed"):
    print("Pre-processing for deeptabular")
    X_tab_train = tab_preprocessor.fit_transform(train)
    X_tab_val = tab_preprocessor.transform(val)
    X_tab_test = tab_preprocessor.transform(test)

    with open("/workspace/dataset_challenge/geo_tab_preprocessed", "wb") as f:
        pickle.dump([X_tab_train, X_tab_val, X_tab_test, tab_preprocessor], f, protocol=4)
else:
    print("Loading pre-processed deeptabular")
    with open("/workspace/dataset_challenge/geo_tab_preprocessed", "rb") as f:
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
# Option 1: this will also save training history and lr history if the
# LRHistory callback is used
trainer.save(path="./geo_model_weights", save_state_dict=True)

# Save Preds
np.save("./geo_preds.npy", preds)