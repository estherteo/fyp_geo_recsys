import pandas as pd
from recommenders.evaluation import python_evaluation as msft_eval

def get_metrics(pred_df_fpath: str, true_df_fpath: str):
    pred_df = pd.read_parquet(pred_df_fpath)
    true_df = pd.read_parquet(true_df_fpath)

    pred_df['user_id'] = pred_df['user_id'].astype(int)
    pred_df['business_id'] = pred_df['business_id'].astype(int)

    true_df['user_id'] = true_df['user_id'].astype(int)
    true_df['business_id'] = true_df['business_id'].astype(int)

    pred_df['prediction'] = pred_df['prediction'].astype(float)
    pred_df['prediction'] = pred_df['prediction'] * 5

    pred_df = pred_df[pred_df['prediction'] >= 3]

    true_df['review_stars'] = true_df['review_stars'].astype(float)

    true_df = true_df[true_df['review_stars'] >= 3]

    print("Getting precision @ 10")
    prec = msft_eval.precision_at_k(true_df, pred_df,
                                col_user='user_id',
                                col_item='business_id',
                                col_rating='review_stars',
                                col_prediction='prediction',
                                relevancy_method='by_threshold',
                                k=10, threshold=3)

    print("Getting recall @ 10")
    recall = msft_eval.recall_at_k(true_df, pred_df,
                                col_user='user_id',
                                col_item='business_id',
                                col_rating='review_stars',
                                col_prediction='prediction',
                                relevancy_method='by_threshold',
                                k=10, threshold=3)

    print("Getting NDCG @ 10")
    ndcg = msft_eval.ndcg_at_k(true_df, pred_df,
                                col_user='user_id',
                                col_item='business_id',
                                col_rating='review_stars',
                                col_prediction='prediction',
                                relevancy_method='by_threshold',
                                k=10, threshold=3)



    print(f"Precision@10: {prec}")
    print(f"Recall@10: {recall}")
    print(f"NDCG@10: {ndcg}")

if __name__ == "__main__":
    get_metrics(pred_df_fpath="/root/data/mlp_predictions.parquet", true_df_fpath="/root/data/yelp_reduced_test.parquet")