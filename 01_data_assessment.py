"""Find 2022 Excellence in Healthcare Winners

:authors: Joe Skvarna
:date: 11/29/2023
"""

# import packages
import pandas as pd
from os import path
import numpy as np

# set file and data paths
DATA_DIR = "/Users/joeskvarna/Desktop/IAH Assessment/"

input_awards = path.join(DATA_DIR, "IHA DA Challenge Instructions/awards_challenge_my2022.csv")
input_measures = path.join(DATA_DIR, "IHA DA Challenge Instructions/measure_list_my2022.csv")

################################################################
# create functions
def merge_validate(df_left, df_right, valid):
    """Creates a merged dataframe on measure_code via left merge. Validates that all records from both df's merged.
    
    Args:
        df_left (df): left df to use for merge
        df_right (df): right df to merge on
        valid (str): assert which kind of merge (i.e. m:1, 1:m, 1:1)
        
    
    Returns:
        A new, merged dataframe
    """
    
    df = pd.merge(df_left, df_right, on='measure_code', how='left', indicator=True, validate=(valid))
    assert (df['_merge'] == 'both').all()
    df = df.drop(columns=['_merge'])
    
    return df

def get_unique_list(df, colname):
    """Creates a list of the unique values in a df's designated column
    
    Args:
        df (df): dataframe to use
        colname (str): column to create unqiue values from
        
    
    Returns:
        A list with unique values
    """
    df_return = list(np.unique(df[colname]))
    return df_return

################################################################
# start task
# 0.1 read in both data files
df_awards = pd.read_csv(input_awards)

df_measures = pd.read_csv(input_measures)
assert df_measures['measure_code'].is_unique

# 0.2 Merge data files together
df_main = merge_validate(df_awards, df_measures, 'm:1')

# assert unique at provider and measure level
is_dup = df_main.duplicated(['po_id', 'measure_code'])
assert is_dup.all() == False



# 1.1 Ensure directionality of rate
# create list of measures where a lower rate is better
df_main_lwr_better = df_main.loc[df_main['higher_is_better'] == False]
lst_lwr_better = get_unique_list(df_main_lwr_better, 'measure_code')
print(lst_lwr_better)

# check useful values to help decide how to change directionality of rate
# double check 50 column
# calculate max rate by measure
df_main['max_measure_rate'] = df_main.groupby('measure_code')['rate'].transform('max')
df_main['min_measure_rate'] = df_main.groupby('measure_code')['rate'].transform('min')
print(df_main[['measure_code','higher_is_better', 'domain', 'max_measure_rate', 'min_measure_rate']].value_counts())

# create a new rates column
df_main = df_main.rename(columns={'rate': 'rate_old'})
df_main['rate'] = df_main['rate_old']

# 3 clinical measures with higher_is_better = False
# set these measures (max_rate <= 100) to 100 - rate
mask = (df_main['higher_is_better'] == False) & (df_main['max_measure_rate'] <= 100)
df_main.loc[mask, 'rate'] = 100 - df_main['rate_old']

# the 1 cost measure has higher_is_better = False
# set this measure (max_rate > 100) to itself * -1
# this is just the money column
mask = (df_main['higher_is_better'] == False) & (df_main['max_measure_rate'] > 100)
df_main.loc[mask, 'rate'] = (-1)*df_main['rate_old']

# compare rate vs rate old
# create a df where rate was changed and where it was not changed
df_changed = df_main[['measure_code', 'rate_old', 'rate', 'higher_is_better']].loc[df_main['rate_old'] != df_main['rate']]
df_same = df_main[['measure_code', 'rate_old', 'rate', 'higher_is_better']].loc[df_main['rate_old'] == df_main['rate']]

# assert the only rates that were changed were where higher_is_better == False
assert df_changed['higher_is_better'].all() == False
assert (df_same['higher_is_better'].all() == True) | ((df_same['rate'] == 50).any())

# assert measures changed matches our measures where lower was better
lst_measures_rate_chng = get_unique_list(df_changed, 'measure_code')
assert lst_lwr_better == lst_measures_rate_chng

# change higher is better to True for all after changing/ensuring directionality
df_main['higher_is_better'] = True

# drop are interim max and min measure rates
df_main = df_main.drop(columns=['max_measure_rate', 'min_measure_rate'])



# 1.2 Flag invalid measure data
# create valid column
df_main['valid'] = 1

# assert rate is never missing and fill in invalid values
assert df_main['rate'].isna().all() == False
mask = (df_main['denominator'] <= 30) | (df_main['reliability'] <= .7)
df_main.loc[mask, 'valid'] = 0 

# summarize valid response column
print(df_main['valid'].describe())
# ~ 85% of records are valid





# 2.1 create min, max, avg rate by measure using only valid records
# create df of only valid records
df_valid = df_main.loc[df_main['valid'] == 1]

# get global avg, min, max by meaure code using valid records
df_valid_measures = df_valid.groupby('measure_code').agg(
    measure_global_avg =('rate', 'mean'),
    measure_global_max =('rate', 'max'),
    measure_global_min = ('rate', 'min'))

# merge min, max, avg global values onto with invalid records
df_main_measures = merge_validate(df_valid, df_valid_measures, 'm:1')

# take only valid records from this new dataset with global measures
df_valid_rates = df_main_measures.loc[df_main_measures['valid'] == 1]

# 2.2 calculate difference between rate and global average at po level using only valid records
df_valid_rates['rate_difference'] = df_valid_rates['rate'] - df_valid_rates['measure_global_avg']

# 2.3 calculate po's avg rate difference by domain using valid records
df_valid_rates['avg_rate_diff_domain'] = df_valid_rates.groupby(['po_id', 'domain'])['rate_difference'].transform('mean')
df_valid_grouped = df_valid_rates[['po_id', 'domain', 'avg_rate_diff_domain']].drop_duplicates()

# merge onto our main dataset to get po's avg rate difference for invalid records
df_main_rates = pd.merge(df_main_measures, df_valid_grouped, on=['po_id', 'domain'], how='left', indicator=True, validate = ('m:1'))
assert (df_main_rates['_merge'] == 'both').all()
df_main_rates = df_main_rates.drop(columns=['_merge'])

# now we have po's avg rate difference for all records but we only used valid records to calculate it!

# 2.4a before imputing rate for invalid records, set invalid records to missing rate
mask = (df_main_rates['valid'] == 0)
df_main_rates.loc[mask,'rate'] = np.nan

# 2.4b calculate the imputed rate for invalid or missing measures using avg_rate_diff_domain
# imputed rate = measure_global_avg + avg_rate_diff_domain for invalid records
mask = (df_main_rates['valid'] == 0)
df_main_rates.loc[mask, 'rate'] = df_main_rates['measure_global_avg'] + df_main_rates['avg_rate_diff_domain']

# no missing values for rate after imputation
assert df_main_rates['rate'].isna().any() == False

# 2.4c correct imputed rates that are out of max, min range
# if imputed rate is greater than global measure max, set to max
mask = (df_main_rates['valid'] == 0) & (df_main_rates['rate'] > df_main_rates['measure_global_max'])
df_main_rates.loc[mask, 'rate'] = df_main_rates['measure_global_max']

# if imputed rate is less than global measure min, set to min
mask = (df_main_rates['valid'] == 0) & (df_main_rates['rate'] < df_main_rates['measure_global_min'])
df_main_rates.loc[mask, 'rate'] = df_main_rates['measure_global_min']

# 2.5 remove invalid po's if they have too many invalid records within a domain
# rule: if more than half of a po's measures in a domain are invalid, the po is not valid for the award
# calculate total count in domain and valid in domain by po
df_main_rates['num_valid_in_domain'] = df_main_rates.groupby(['po_id', 'domain'])['valid'].transform('sum')
df_main_rates['domain_total_obs'] = df_main_rates.groupby(['po_id', 'domain'])['domain'].transform('count')

df_main_rates[['domain', 'num_valid_in_domain', 'domain_total_obs']].value_counts()

# remove records where this ratio is less than 1/2 BUT
# if exact 1/2 fraction isn't possible  (i.e. 7/15 has no opportunity for exactly 1/2), we round up to 1/2
# to accomplish this, I will substract 1 from the denominator when the denominator is odd
# I won't do this when the numerator is 0
mask = (df_main_rates['domain_total_obs'] % 2 != 0) & (df_main_rates['num_valid_in_domain'] != 0)
df_main_rates.loc[mask, 'domain_total_obs'] = df_main_rates['domain_total_obs'] - 1

# now, I can cut off at .5 (50% valid)
df_main_rates['valid_ratio'] = df_main_rates['num_valid_in_domain']/df_main_rates['domain_total_obs']
df_final = df_main_rates[df_main_rates['valid_ratio'] >= .5]



# 3.0 Now all rates are valid and we are only working with valid po's
# 3.1 calculate composite score: avg rate by po and domain
df_final['composite_score'] = df_final.groupby(['po_id', 'domain'])['rate'].transform('mean')

# 3.2 calculate global median of composite score by domain
df_final['domain_median'] = df_final.groupby(['domain'])['composite_score'].transform('median')

# 3.3: calculate median for total cost of care
# Note: 
# I included this in the domains above and in the prior imputations
# Cost is also already onversed to negative, so a higher cost is better

# 3.4 winners score better or equal to the median in all 3 domains
# remove records where composite score is below the median
mask = df_final['composite_score'] >= df_final['domain_median']
df_winner = df_final.loc[mask]

# only keep po's that have all 3 domains in the dataset
df_winner['num_domains'] = df_winner.groupby('po_id')['domain'].transform('nunique')
df_winner = df_winner[df_winner['num_domains'] == 3]

# drop duplicates since we just want org name and po
df_winners_unique = df_winner[['org_name', 'po_id']].drop_duplicates()
df_winners_unique = df_winners_unique.sort_values('org_name')

# output winners to txt file
num_winners = len(df_winners_unique)
num_total = len(df_awards[['org_name', 'po_id']].drop_duplicates())

with open(path.join(DATA_DIR, "submission_iah_winners.txt"), 'w') as f:
    f.write(f"{num_winners}/{num_total} organizations are Excellence in Healthcare Winners in 2022!\n")
    f.write("The list of winners in alphabetical order are below: \n \n")
    str_df= df_winners_unique.to_string(header=True, index=False)
    f.write(str_df)

