import pandas as pd


def train_test_split(train_df,test_len):
    train_df.dropna(inplace=True) # error
    train_df = train_df[train_df['ITEM_CD']=='1000043']        
    test_output = train_df[-test_len:]
    train_output = train_df[:-test_len]
    return train_output,test_output

# del features and one-hot encoding
def common_processing(df):
    df = df.set_index('ISO_YEAR_WEEK')    
    del df['ITEM_CD']
    df_dtypes = df.dtypes
    # one-hot encoding
    to_one_hot_columns = list(df_dtypes[df_dtypes == 'object'].index)    
    to_one_hot = df[to_one_hot_columns]
    one_hot = pd.get_dummies(to_one_hot)
    df.drop(columns=to_one_hot_columns, inplace=True)
    result = pd.concat([df,one_hot],axis=1)
    return result

def delete_unique(train_df):
    for col in train_df.columns:
        if len(pd.unique(train_df[col]))==1:
            del train_df[col]
    return train_df

def match_col_train_test(train,test):
    train_columns = train.columns
    missing_cols = train_columns.difference(test.columns)
    # Add a missing column in test set with default value equal to 0
    for col in missing_cols:
        test[col] = 0
    # Ensure the order of column in the test set is in the same order than in train set
    test = test[train_columns]
    return test