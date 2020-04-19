import pandas as pd
import os
import sys


def extract_data(messages_filepath, categories_filepath):
    """Load data
    Returns:
        message and category dataframes
    """
    msg_df = pd.read_csv(messages_filepath)
    cat_df = pd.read_csv(categories_filepath)

    return msg_df, cat_df


def clean_data(msg_df, cat_df):
    """Clean and combine data
    Returns:
        Deduplicated and merged dataframe
    """
    # Remove duplicates
    for df in [msg_df, cat_df]:
        dupes = df.id.duplicated()
        df.drop(df.index[dupes], inplace=True)
        df.set_index('id', inplace=True)
        print(f"Data has shape: {df.shape}")

    # Merge data
    df_out = msg_df.merge(cat_df, left_index=True, right_index=True)
    return df_out


def prep_data(df):
    """Modify data to prepare for analysis
    Returns:
        Dataframe with useful features
    """

    # Make feature for translated messages
    eng_msg = (df.message == df.original) | (df.original.isna())
    df['translated'] = eng_msg

    # Split message categories (convert into integers)
    cat_cols = [col[:-2] for col in df.loc[2, 'categories'].split(';')]
    cat_str = df.categories.replace(r'[^012;]', '', regex=True)
    cat_vals = cat_str.str.split(';', expand=True).astype('int')
    cat_vals.columns = cat_cols
    # Remove labels with no instances
    for col in cat_vals.columns:
        if cat_vals[col].sum() == 0:
            print(f"Dropping {col} because it has no instances")
            cat_vals.drop(columns=col, inplace=True)

    # Join message categories to table
    df = pd.concat([df, cat_vals], axis=1, sort=False)
    df.drop(columns=['original', 'categories'], inplace=True)

    return df


def save_data(df, database_filepath, tn='scored_messages'):
    """Save table into SQLite database"""
    from sqlalchemy import create_engine
    engine = create_engine(f'sqlite:///{database_filepath}')

    if engine.dialect.has_table(engine, tn):
        from sqlalchemy import MetaData, Table
        meta = MetaData()
        tbl = Table(tn, meta)
        tbl.drop(engine)

    df.to_sql(tn, engine, index=False)


def main():
    print(os.getcwd())
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print("Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}"
              .format(messages_filepath, categories_filepath))
        msg_df, cat_df = extract_data(messages_filepath, categories_filepath)

        print("Cleaning data...")
        df = clean_data(msg_df, cat_df)

        print("Prepping data")
        df = prep_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
