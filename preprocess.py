import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/Ischemia.txt', sep='\t')
df = df.dropna()

df['Ischemia'], _ = pd.factorize(df['Ischemia'])
df['Diastolic.function'], _ = pd.factorize(df['Diastolic.function'])
df['age'] -= df['age'].min()
df['age'] /= df['age'].max()

df = df.sample(frac=1).reset_index(drop=True)
data = df

# df.plot(x='Ischemia', y='Diastolic.function', kind='scatter')
df.plot.scatter('A', 'E', c='Ischemia', colormap='jet')

# %%

nrow = df.shape[0]
n_test = round(nrow * 0.2)
n_val = round(nrow * 0.1)


def get_test_data():
    df = data.iloc[:n_test]
    y = df['Ischemia'].to_numpy().astype(int)
    x = df.drop('Ischemia', axis=1).to_numpy()
    return x, y


def get_val_data():
    df = data.iloc[n_test:n_test + n_val]
    y = df['Ischemia'].to_numpy().astype(int)
    x = df.drop('Ischemia', axis=1).to_numpy()
    return x, y


def get_train_data():
    df = data.iloc[n_test + n_val:nrow]
    y = df['Ischemia'].to_numpy().astype(int)
    x = df.drop('Ischemia', axis=1).to_numpy()
    return x, y
