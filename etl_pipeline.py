"""
ETL pipeline for disaster recovery exercise

Assumes data are locally available
"""

# Libraries
import pandas as pd
from sqlalchemy import create_engine

# Load data
messages = pd.read_csv("data/messages.csv")
categories = pd.read_csv("data/categories.csv")

# Remove duplicates
m_dupes = messages.id.duplicated()
messages.drop(messages.index[m_dupes], inplace=True)
messages.set_index('id', inplace=True)
print(f"Messages data loaded with shape: {messages.shape}")

c_dupes = categories.id.duplicated()
categories.drop(categories.index[c_dupes], inplace=True)
categories.set_index('id', inplace=True)
print(f"Category data loaded with shape: {messages.shape}")

# Merge data
df = messages.merge(categories, left_index=True, right_index=True)

# Split message categories
cat_cols = [col[:-2] for col in df.loc[2, 'categories'].split(';')]
cat_str = df.categories.replace(r'[^01;]', '', regex=True)
cat_vals = cat_str.str.split(';', expand=True)
cat_vals.columns = cat_cols

# Join message categories to table
df = pd.concat([df, cat_vals], axis=1, sort=False)
df.drop(columns=['original', 'categories'], inplace=True)

# Insert into database (clearing out table if it already exists)
engine = create_engine('sqlite:///data/DisasterResponse.db')
tn = 'scored_messages'

if engine.dialect.has_table(engine, tn):
    from sqlalchemy import MetaData, Table
    meta = MetaData()
    tbl = Table('tn', meta)
    tbl.drop(engine)

df.to_sql(tn, engine, index=False)
