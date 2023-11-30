"""Find 2022 Excellence in Healthcare Winner

:authors: Joe Skvarna
:date: 11/29/2023
"""

# import packages
import pandas as pd
from os import path
import numpy as np

# set file and data paths
main = "/Users/joeskvarna/Desktop/IAH Assessment/"

input_awards = path.join(main, "IHA DA Challenge Instructions/awards_challenge_my2022.csv")
input_measures = path.join(main, "IHA DA Challenge Instructions/measure_list_my2022.csv")

################################################################
# create functions
def merge_validate(df_left, df_right, valid):
    """Creates a merged dataframe on measure code via left merge. Validates that all records from both df's merged.
    
    Args:
        df_left (df): left df to use for merge
        df_right (df): right df to merge on
        valid (str): assert which kind of merge (i.e. m:1, 1:m, 1:1)
        
    
    Returns:
        A new dataframe with given name
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
# 0.1 Read in data files
df_awards = pd.read_csv(input_awards)
df_measures = pd.read_csv(input_measures)

# assert measure is unique in measures data
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

# use measure_list.csv and double check with max rate by measure to see whether a measure is on a 0-100 percent scale or is continuous (money column)
# calculate max rate by measure
df_main['max_measure_rate'] = df_main.groupby('measure_code')['rate'].transform('max')

# create a new rates column
df_main = df_main.rename(columns={'rate': 'rate_old'})
df_main['rate'] = df_main['rate_old']

# set percent measures (max_rate <= 100) to 100 - rate
mask = (df_main['higher_is_better'] == False) & (df_main['max_measure_rate'] <= 100)
df_main.loc[mask, 'rate'] = 100 - df_main['rate_old']

# set continous measures (max_rate > 100) to itself * -1
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

# assert measures changed matches our measures where higher is worse
lst_measures_rate_chng = get_unique_list(df_changed, 'measure_code')
assert lst_lwr_better == lst_measures_rate_chng

# change higher is better to True for all after ensuring directionality
df_main['higher_is_better'] = True
df_main = df_main.drop(['max_measure_rate'], axis = 1)


# 1.2 Flag invalid measure data
# create valid column
df_main['valid'] = 1

# create invalid condition and assert rate is never missing
mask = (df_main['denominator'] <= 30) | (df_main['reliability'] <= .7)
assert df_main['rate'].isna().all() == False
df_main.loc[mask, 'valid'] = 0 

# summarize valid response column
df_main['valid'].describe()




# 2.1 create min, max, avg rate by measure using only valid records
df_valid = df_main.loc[df_main['valid'] == 1]

df_valid_measures = df_valid.groupby('measure_code').agg(
    measure_global_avg =('rate', 'mean'),
    measure_global_max =('rate', 'max'),
    measure_global_min = ('rate', 'min'))

# merge min, max, avg values back onto main dataframe
df_main_rates = merge_validate(df_main, df_valid_measures, 'm:1')

# summarize avg_measure_rate by measure
df_main_rates[['measure_code', 'measure_global_avg']].value_counts()

# 2.2 calculate difference between rate and global average at po level
# FLAG: should I put rate and rate_difference as missing up here first? I want to treat the transform below to skip over missings
df_main_rates['rate_difference'] = df_main_rates['rate'] - df_main_rates['measure_global_avg']

# 2.3 calculate po's avg rate difference for each domain (clinical quality and patient experiene)
# calculate average difference by provider and domain
df_main_rates['avg_rate_diff_domain'] = df_main_rates.groupby(['po_id', 'domain'])['rate_difference'].transform('mean')

# take a look at the unique df
df_unique = df_main_rates[['po_id', 'domain', 'avg_rate_diff_domain']].drop_duplicates()
df_unique.sort_values('po_id')

# set rate to missing if it wasn't valid
mask = (df_main['valid'] == 0)
df_main_rates.loc[mask,'rate'] = np.nan

# 2.4 calculate the imputed rate for invalid or missing measures using the POʼs Average Measure Rate Difference
# fill in where rate_diff is missing
# imputed rate = measure_global_avg + avg_rate_diff_domain
mask = (df_main_rates['valid'] == 0)
df_main_rates.loc[mask, 'rate'] = df_main_rates['measure_global_avg'] + df_main_rates['avg_rate_diff_domain']

assert df_main_rates['rate'].isna().any() == False

# if imputed rate is greater than global measure max, set to max
mask = (df_main_rates['valid'] == 0) & (df_main_rates['rate'] > df_main_rates['measure_global_max'])
df_main_rates.loc[mask, 'rate'] = df_main_rates['measure_global_max']

# same idea for min
mask = (df_main_rates['valid'] == 0) & (df_main_rates['rate'] < df_main_rates['measure_global_min'])
df_main_rates.loc[mask, 'rate'] = df_main_rates['measure_global_min']

# 2.5 if more than half of a Po's measures in a domain are invalid, the po is not valid for the award
# calculate total count in domain and valid in domain by po
df_main_rates['num_valid_in_domain'] = df_main_rates.groupby(['po_id', 'domain'])['valid'].transform('sum')
df_main_rates['domain_total_obs'] = df_main_rates.groupby(['po_id', 'domain'])['domain'].transform('count')

# want to remove records where this ratio is 1/2 BUT
# if a half score is 7/15 (i.e. no opportunity for percet even), we round up to pass
# to accomplish this, I will substract 1 from the denominator when the denominator is odd
mask = df_main_rates['domain_total_obs'] % 2 != 1
df_main_rates.loc[mask, 'domain_total_obs'] = df_main_rates['domain_total_obs'] - 1

# now, I can cut off at .5 (50% valid)
df_main_rates['valid_ratio'] = df_main_rates['num_valid_in_domain']/df_main_rates['domain_total_obs']
df_final = df_main_rates[df_main_rates['valid_ratio'] >= .5]



# 3.0 Now all rates are valid and we are only working with valid po's
# 3.1 calculate composite score - avg rate by po and domain
df_final['composite_score'] = df_final.groupby(['po_id', 'domain'])['rate'].transform('mean')

# 3.2 calculate global median of composite score by domain
df_final['domain_median'] = df_final.groupby(['domain'])['composite_score'].transform('median')

# 3.3: calculate median for total cost of care
# we already did this above
# also, I included cost in all of our prior imputations
# I also already inversed the cost to be negative, so a higher cost will now be better

# 3.4 winners score better or equal to the median in all 3 domains
# remove records where composite score is below the median
mask = df_final['composite_score'] >= df_final['domain_median']
df_winner = df_final.loc[mask]

# only keep po's that have all 3 domains in the dataset
df_winner['unique_count'] = df_winner.groupby('po_id')['domain'].transform('nunique')
df_winner = df_winner[df_winner['unique_count'] == 3]

# drop duplicates since we just want org name and po
df_winners_unique = df_winner[['org_name', 'po_id']].drop_duplicates()

num_winners = len(df_winners_unique)
num_total = len(df_awards[['org_name', 'po_id']].drop_duplicates())

print(f"{num_winners}/{num_total} organizations are Excellence in Healthcare Winners in 2022!\n")
print("The list of winners is below: \n")
print(df_winners_unique)

