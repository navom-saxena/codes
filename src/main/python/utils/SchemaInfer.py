import warnings

import pandas as pd
from pydqc import infer_schema

warnings.filterwarnings('ignore')

path = "/Users/navomsaxena/Downloads/titanic.csv"
outputpath = "/Users/navomsaxena/Downloads/"

df = pd.read_csv(path, sep=',', low_memory=False)

infer_schema.infer_schema(data=df, fname='properties', output_root=outputpath,
                          sample_size=1.0, type_threshold=0.5, n_jobs=1,
                          base_schema=None)

print("data")
df.head()