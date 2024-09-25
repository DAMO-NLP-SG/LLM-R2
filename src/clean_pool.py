

import pandas as pd


def clean_pool(poll_csv):
    pool_df = pd.read_csv(poll_csv)
    clean_df = pool_df.drop_duplicates(subset=['activated_rules', 'original_logical_plan']).reset_index()
    return clean_df


dataset = 'dsb'
promo_pool_pos = clean_pool('pos_pool_' + dataset + '_updated.csv')
promo_pool_neg = clean_pool('neg_pool_' + dataset + '_updated.csv')

promo_pool_pos.to_csv('pos_pool_' + dataset + '_cleaned.csv')
promo_pool_neg.to_csv('neg_pool_' + dataset + '_cleaned.csv')
# print(promo_pool_pos)
