import pandas as pd
from utils import word2char


# error_types = ['Cognitive Error', 'Homonym Error', 'Run-on Error',
#      'Split-word Error (Left)', 'Split-word Error (Random)',
#      'Split-word Error (Right)', 'Split-word Error (both)',
#      'Typo (Avro) Substituition', 'Typo (Bijoy) Substituition',
#      'Typo Deletion', 'Typo Insertion', 'Typo Transposition',
#      'Visual Error', 'Visual Error (Combined Character)']


def error_df(df, error='Cognitive Error'):
    df = df.loc[df['ErrorType'] == error]
    df['Word'] = df['Word'].apply(word2char)
    df['Error'] = df['Error'].apply(word2char)
    df = df.sample(frac=1).reset_index(drop=True)
    df = df.iloc[:, [1, 0]]
    df.to_csv('./Dataset/error.csv', index=False)

