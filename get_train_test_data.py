import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import apply_all_transforms, normalize_df
import warnings
warnings.filterwarnings('ignore')

SEED = 42
TEST_SIZE = 0.15
# delete values greater than DEL_THR
DEL_THR = 60000

df = pd.read_csv('data/LTV.csv')
del df['Customer'], df['Effective To Date']

df = apply_all_transforms(df)

# delete rare extremely high values
y = df.pop('Customer Lifetime Value').values
df = df.loc[y < DEL_THR]
y = y[y < DEL_THR]

df = normalize_df(df)
df['Customer Lifetime Value'] = y

tr_df, te_df = train_test_split(df, test_size=TEST_SIZE, random_state=SEED)

tr_df.to_csv('data/train.csv', index=False)
te_df.to_csv('data/test.csv', index=False)
