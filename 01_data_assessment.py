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
def merge_validate(df_left, df_right, valid, onto):
    """Creates a merged dataframe via left merge. Validates that all records from both df's merged.
    
    Args:
        df_left (df): left df to use for merge
        df_right (df): right df to merge on
        valid (str): assert which kind of merge (m:1, 1:m, 1:1)
        onto (str, lst): column(s) to merge onto
        
    
    Returns:
        A new, merged dataframe
    """
    
    df = pd.merge(df_left, df_right, on=onto, how='left', indicator=True, validate=(valid))
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
if __name__ == '__main__':
    
    # read in both files
    df_awards = pd.read_csv(input_awards)
    df_measures = pd.read_csv(input_measures)
    assert df_measures['measure_code'].is_unique
    
    # merge together
    df_main = merge_validate(df_awards, df_measures, 'm:1', 'measure_code')
    
    # assert file unique at po & measure
    is_dup = df_main.duplicated(['po_id', 'measure_code'])
    assert is_dup.all() == False
    
    
    ##########################################
    # 1.1 Ensure directionality of rate
    ##########################################
    # list where lower rate is better
    df_main_lwr_better = df_main.loc[df_main['higher_is_better'] == False]
    lst_lwr_better = get_unique_list(df_main_lwr_better, 'measure_code')
    print(f"Measures where lower rate is better: {lst_lwr_better}\n")
    
    # calculate max, min of rate by measure to see scale of measures
    # note: I also looked at the measure_list.csv descriptions to see scale
    df_main['max_measure_rate'] = df_main.groupby('measure_code')['rate'].transform('max')
    df_main['min_measure_rate'] = df_main.groupby('measure_code')['rate'].transform('min')
    #print(df_main[['measure_code','higher_is_better', 'domain', 'max_measure_rate', 'min_measure_rate']].value_counts())
    
    # create a new rates column
    df_main = df_main.rename(columns={'rate': 'rate_old'})
    df_main['rate'] = df_main['rate_old']
    
    # set clinical measures (percent measures, 0-100, max rate <= 100) to 100 - rate
    mask = (df_main['higher_is_better'] == False) & (df_main['max_measure_rate'] <= 100)
    df_main.loc[mask, 'rate'] = 100 - df_main['rate_old']

    # set cost measure (continuous, max_rate > 100) to itself * -1
    mask = (df_main['higher_is_better'] == False) & (df_main['max_measure_rate'] > 100)
    df_main.loc[mask, 'rate'] = (-1)*df_main['rate_old']
    
    ####
    # QA directionality
    ####
    # create a df where rate was changed and where it was not changed
    df_changed = df_main[['measure_code', 'rate_old', 'rate', 'higher_is_better']].loc[df_main['rate_old'] != df_main['rate']]
    df_same = df_main[['measure_code', 'rate_old', 'rate', 'higher_is_better']].loc[df_main['rate_old'] == df_main['rate']]
    
    # assert the only rates that were changed were where higher_is_better == False
    assert df_changed['higher_is_better'].all() == False
    assert (df_same['higher_is_better'].all() == True) | ((df_same['rate'] == 50).any())
    
    # assert measures changed matches our measures where lower was better
    lst_measures_rate_chng = get_unique_list(df_changed, 'measure_code')
    assert lst_lwr_better == lst_measures_rate_chng
    
    # change higher is better to True for all after QA
    df_main['higher_is_better'] = True
    
    # drop max and min measure rates after QA
    df_main = df_main.drop(columns=['max_measure_rate', 'min_measure_rate'])
    ###
    
    ##########################################
    # 1.2 Flag invalid measure data
    ##########################################
    # create valid column
    df_main['valid'] = 1
    
    # fill in invalid values
    assert df_main['rate'].isna().any() == False
    assert df_main['measure_code'].isna().any() == False
    mask = ((df_main['denominator'] <= 30) & (df_main['domain'] == "Clinical")) | ((df_main['reliability'] <= .7) & (df_main['domain'] == "Patient Experience"))
    df_main.loc[mask, 'valid'] = 0 
    
    # summarize valid response column
    print("Description of valid responses:")
    print(df_main['valid'].describe())
    # ~ 85% of records are valid
    
    
    print("\nstep 1/4 complete \n")
    ##########################################
    # 2.1 Create min, max, avg rate by measure using only valid records
    ##########################################
    df_valid = df_main.loc[df_main['valid'] == 1]

    # calculate global avg, min, max by meaure using only valid records
    df_valid_measures = df_valid.groupby('measure_code').agg(
        measure_global_avg =('rate', 'mean'),
        measure_global_max =('rate', 'max'),
        measure_global_min = ('rate', 'min'))
    
    # merge min, max, avg global values onto main df with invalid records
    df_main = merge_validate(df_main, df_valid_measures, 'm:1', 'measure_code')
    
    ##########################################
    # 2.2 Calculate difference between rate and global average at po level using only valid records
    ##########################################
    df_main['rate_difference'] = df_main['rate'] - df_main['measure_global_avg']
    
    ###
    # NOTE: I was confused on the language of 2.2 and 2.3 in the instructions
    # I made a note in my submission
    # lines commented out (prefix 1-6) calculate avg rate difference only using valid rates
    
    # use only valid records of current df
    # 1. df_valid = df_main.loc[df_main['valid'] == 1]
    
    # calculate rate difference
    # 2. df_valid['rate_difference'] = df_valid['rate'] - df_valid['measure_global_avg']
    ###
    ##########################################
    # 2.3 Calculate avg rate difference by po, domain
    ##########################################
    df_main['avg_rate_diff_domain'] = df_main.groupby(['po_id', 'domain'])['rate_difference'].transform('mean')
    
    ###
    # Other approach commented out
    # 3. df_valid['avg_rate_diff_domain'] = df_valid.groupby(['po_id', 'domain'])['rate_difference'].transform('mean')
    # 4. df_valid_grouped = df_valid[['po_id', 'domain', 'avg_rate_diff_domain']].drop_duplicates()
    
    # merge onto our main dataset
    # this get's po's avg rate difference for invalid records
    # 5. df_main = merge_validate(df_main, df_valid_grouped, onto=['po_id', 'domain'], valid=None)
    # 6. df_main = pd.merge(df_main, df_valid_grouped, on=['po_id', 'domain'], how='left')
    
    # now we have po's avg rate difference for all records but we only used valid records to calculate it!
    ###
    ##########################################
    # 2.4a Calculate imputed rate for invalid records
    #########################################
    # set invalid records to missing rate
    mask = (df_main['valid'] == 0)
    df_main.loc[mask,'rate'] = np.nan
    
    # calculate the imputed rate for invalid or missing measures using measure_global_avg + avg_rate_diff_domain
    mask = (df_main['valid'] == 0)
    df_main.loc[mask, 'rate'] = df_main['measure_global_avg'] + df_main['avg_rate_diff_domain']

    assert df_main['rate'].isna().any() == False
    
    ##########################################
    # 2.4b Correct imputed rates that are out of global max, min range
    ##########################################
    # if imputed rate is greater than global measure max, set to max
    mask = (df_main['valid'] == 0) & (df_main['rate'] > df_main['measure_global_max'])
    df_main.loc[mask, 'rate'] = df_main['measure_global_max']
    
    # if imputed rate is less than global measure min, set to min
    mask = (df_main['valid'] == 0) & (df_main['rate'] < df_main['measure_global_min'])
    df_main.loc[mask, 'rate'] = df_main['measure_global_min']
    
    ##########################################
    # 2.5 If more than half of a po's measures in a domain are invalid, the po is not valid for the award
    ##########################################
    # calculate total count in domain and total valid in domain by po
    df_main['num_valid_in_domain'] = df_main.groupby(['po_id', 'domain'])['valid'].transform('sum')
    df_main['domain_total_obs'] = df_main.groupby(['po_id', 'domain'])['domain'].transform('count')
    
    # remove records where this ratio is less than 1/2 BUT
    # if exact 1/2 fraction isn't possible  (i.e. 7/15 has no opportunity for exactly 1/2), we round up to 50% and pass
    # to accomplish this, I will substract 1 from the denominator when the denominator is odd
    # I won't do this when the numerator is 0
    mask = (df_main['domain_total_obs'] % 2 != 0) & (df_main['num_valid_in_domain'] != 0)
    df_main.loc[mask, 'domain_total_obs'] = df_main['domain_total_obs'] - 1
    
    # now, cut off at .5 (50% valid)
    df_main['valid_ratio'] = df_main['num_valid_in_domain']/df_main['domain_total_obs']
    df_main = df_main[df_main['valid_ratio'] >= .5]
    
    
    print("step 2/4 complete \n")
    
    ##########################################
    # 3.0 Declare Winner
    # all rates are now imputed correctly and we are only working with valid po's
    # a winner is better or equal to the median in all 3 domains
    ##########################################
    # 3.1 Calculate composite score: avg rate by po and domain
    df_main['composite_score'] = df_main.groupby(['po_id', 'domain'])['rate'].transform('mean')
    
    # 3.2 Calculate global median of composite score by domain
    df_main['domain_median'] = df_main.groupby(['domain'])['composite_score'].transform('median')
    
    print("Domain composite score medians:")
    print(df_main[['domain', 'domain_median']].drop_duplicates())
    
    # 3.3 Note on: calculating median for total cost of care
    # I included cost in the domains above and in the prior imputations
    # Cost is also already inversed to negative, so a higher cost is better
    # I can proceed a usual with cost
    
    # 3.4 Only keep winners
    # remove records where composite score is below the median
    mask = df_main['composite_score'] >= df_main['domain_median']
    df_main = df_main.loc[mask]
    
    # only keep po's that have all 3 domains above the media
    df_main['num_domains'] = df_main.groupby('po_id')['domain'].transform('nunique')
    df_main = df_main[df_main['num_domains'] == 3]
    
    # keep unique organization and po
    df_winners = df_main[['org_name', 'po_id']].drop_duplicates()
    df_winners = df_winners.sort_values('org_name')
    
    print("\nstep 3/4 complete \n")
    
    # 4.1 Output results
    # output winners to txt file
    num_winners = len(df_winners)
    num_total = len(df_awards[['org_name', 'po_id']].drop_duplicates())
    
    with open(path.join(DATA_DIR, "submission_iah_winners.txt"), 'w') as f:
        f.write(f"{num_winners}/{num_total} organizations are Excellence in Healthcare Winners in 2022!\n")
        f.write("The list of winners in alphabetical order are below: \n \n")
        str_df= df_winners.to_string(header=True, index=False)
        f.write(str_df)
    
    print("step 4/4 complete \n")
    print(f"{num_winners} total winners")
    print("program finished")
