import random
import geopy.distance
from geopy.point import Point
import pandas as pd
from tqdm import tqdm
import math


def calculate_initial_compass_bearing(start, end):
    if (type(start) != tuple) or (type(end) != tuple):
        raise TypeError("Only tuples are supported as arguments")

    lat1 = math.radians(start[0])
    lat2 = math.radians(end[0])

    diffLong = math.radians(end[1] - start[1])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
                                           * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)

    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing


def get_geodesic_midpoint(start_pt, end_pt):
    dist = geopy.distance.geodesic(start_pt, end_pt).km

    bearing = calculate_initial_compass_bearing(start_pt, end_pt)

    midpoint = geopy.distance.distance(
        kilometers=dist
    ).destination(
        start_pt,
        bearing=bearing
    )
    return midpoint


def compute_dist(bbox, lat, lon):
    min_lat, min_lon, max_lat, max_lon = bbox

    bottom_left = (min_lat, min_lon)
    bottom_right = (min_lat, max_lon)
    top_left = (max_lat, min_lon)

    height_midpoint = Point(get_geodesic_midpoint(bottom_left, top_left))  # get the lat
    # print('height_midpoint:', height_midpoint)
    width_midpoint = Point(get_geodesic_midpoint(bottom_left, bottom_right))  # get the lon
    # print('width_midpoint:', width_midpoint)
    return geopy.distance.geodesic((height_midpoint.latitude, width_midpoint.longitude), (lat, lon)).km


class YelpPreProcessor:
    def __init__(self, merged_df_fpath: str, num_neg: int):
        self.df = pd.read_parquet(merged_df_fpath)
        print("Loaded merged df...")

        # Sort self.df on review_date
        self.df = self.df.sort_values(by="review_date")

        print("df columns:", self.df.columns)

        # self.review_cols = [col for col in self.df if col.startswith('review_') and col != 'review_id' and col != 'review_date' and col != 'review_dist_to_centroid' and col != 'review_bboxes']

        # print("self.review_cols:", self.review_cols)

        self.minimal_df = self.df[
            [
                "business_id",
                'business_latitude',
                'business_longitude',
                "review_bboxes",
                "user_id",
                "review_date",
                "review_id",
                "review_bbox_area",
                "review_min_bbox_area",
                "review_area_ratio",
                "review_dist_to_centroid",
                "review_min_bboxes",
            ]
        ]

        # print('Minimal DF Cols:', self.minimal_df.columns)

        # User keys
        user_keys = [key for key in self.df.columns if key.startswith("user_")]
        print("User Keys:", user_keys)
        self.user_df = self.df.drop_duplicates(subset=["user_id"])[user_keys]

        # Business keys
        business_keys = [key for key in self.df.columns if key.startswith("business_")]
        print("Business Keys:", business_keys)
        self.business_df = self.df.drop_duplicates(subset=["business_id"])[
            business_keys
        ]

        # Get unique businesses from self.df
        self.business_minimal_df = self.df.drop_duplicates(subset=["business_id"])[
            ["business_id", "business_latitude", "business_longitude"]
        ]
        self.num_neg = num_neg

    def process(self, output_fpath: str, is_sequential: bool = False):
        neg_samples = []
        userids = []
        review_dates = []
        review_bboxes = []
        review_ids = []
        review_min_area = []
        gen_area = []
        area_ratio = []
        review_min_bboxes = []
        user_grouped = self.minimal_df.groupby("user_id")

        user_neg_already_sampled = {user_id: [] for user_id in self.minimal_df["user_id"].unique()}

        for user_id, user_df in tqdm(user_grouped, total=len(user_grouped)):
            # print("User_df columns:", user_df.columns)
            idx = 0
            user_biz_lst = user_df["business_id"].to_list()
            for _, row in user_df.iterrows():
                if is_sequential:
                    # Is negative if existing interactions are future
                    neg_lst = user_biz_lst[idx:]
                else:
                    neg_lst = []

                # Is also negative if non patronized businesses are in same bbox
                min_lat, min_lon, max_lat, max_lon = row["review_bboxes"]
                mask = (
                    (self.business_minimal_df["business_latitude"] >= min_lat)
                    & (self.business_minimal_df["business_latitude"] <= max_lat)
                    & (self.business_minimal_df["business_longitude"] >= min_lon)
                    & (self.business_minimal_df["business_longitude"] <= max_lon)
                )

                additional_neg_bizid_lst = self.business_minimal_df.loc[
                    mask, "business_id"
                ].tolist()

                if is_sequential:
                    # Get Positive Samples are those that are patronized before the current review date
                    pos_lst = user_biz_lst[:idx]
                else:
                    # Get Positive Samples are those that are patronized (regardless of review date)
                    pos_lst = user_biz_lst

                # Join neg samples and exclude positive samples from negative samples
                neg_lst = list(set(neg_lst + additional_neg_bizid_lst) - set(pos_lst) - set(user_neg_already_sampled[user_id]))

                # Negative sample
                if len(neg_lst) > 0:
                    num_sample = min(len(neg_lst), self.num_neg)

                    if len(neg_lst) >= self.num_neg:
                        choices = random.sample(neg_lst, k=self.num_neg)
                    else:
                        choices = random.choices(neg_lst, k=len(neg_lst))

                    neg_samples.extend(choices)
                    user_neg_already_sampled[user_id].extend(choices)
                    userids.extend([row["user_id"] for _ in range(num_sample)])
                    review_dates.extend([row["review_date"] for _ in range(num_sample)])
                    review_bboxes.extend(
                        [row["review_bboxes"] for _ in range(num_sample)]
                    )
                    review_ids.extend([row["review_id"] for _ in range(num_sample)])
                    gen_area.extend(row["review_bbox_area"] for _ in range(num_sample))
                    review_min_area.extend(
                        row["review_min_bbox_area"] for _ in range(num_sample)
                    )
                    area_ratio.extend(
                        row["review_area_ratio"] for _ in range(num_sample)
                    )
                    review_min_bboxes.extend(
                        row["review_min_bboxes"] for _ in range(num_sample)
                    )
                idx += 1

        print("Constructing Negative Samples DataFrame...")

        neg_df = pd.DataFrame(
            {
                "user_id": userids,
                "business_id": neg_samples,
                "review_date": review_dates,
                "review_id": review_ids,
                "review_bboxes": review_bboxes,
                "review_bbox_area": gen_area,
                "review_min_bbox_area": review_min_area,
                "review_area_ratio": area_ratio,
                "review_min_bboxes": review_min_bboxes,
            }
        )

        print("Joining Negative Samples with Business DataFrame...")
        neg_df = neg_df.merge(self.business_df, on="business_id", how="inner")

        print("Joining Negative Samples with User DataFrame...")
        neg_df = neg_df.merge(self.user_df, on="user_id", how="inner")

        print("Getting review dist")
        neg_df["review_dist_to_centroid"] = neg_df.apply(
            lambda x: compute_dist(x["review_min_bboxes"], x["business_latitude"], x["business_longitude"]),
            axis=1,
        )

        print("Initializing Interaction Columns...")
        # Add the interaction columns
        for col in ["review_stars", 'review_visited']:
            neg_df[col] = 0

        neg_df.to_parquet("/root/neg_df.parquet")

        print("Concatenating Positive and Negative Samples...")
        self.df = pd.concat([self.df, neg_df])

        print("Saving Processed DataFrame...")
        self.df.to_parquet(output_fpath)



if __name__ == '__main__':
    preprocessor = YelpPreProcessor('./dataset_challenge/merged_positives.parquet', 1)
    preprocessor.process('./dataset_challenge/merged_neg_sampled.parquet')