import requests
import pandas as pd
import pygal
from datetime import date
import seaborn as sns
import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt,mpld3
from datetime import date
import numpy as np

pd.set_option('display.max_colwidth', None)  # die komplette Spalte anzeigen
pd.set_option('display.max_columns', None)  # alle Spalten anzeigen
# pd.set_option('display.max_rows', None)

df = pd.read_csv('identitaeten.txt', sep='\t')

# columns setzen
df.columns = ['jahrgang', 'geschlecht', 'count']

df['alter'] = date.today().year - df['jahrgang']

range = pd.Series(np.arange(3))

df['count'] = df['count'].map(lambda x: pd.Series(np.arange(x)))

df = df.explode('count').reset_index()

df.loc[df.geschlecht == 1, "geschlecht"] = "männlich"
df.loc[df.geschlecht == 2, "geschlecht"] = "weiblich"
df.loc[df.geschlecht == 3, "geschlecht"] = "unbekannt"

ax = sns.boxplot(x="geschlecht", y="alter", data=df,
                 order=["männlich", "weiblich", "unbekannt"])

plt.xlabel("")
plt.ylabel("Alter")
plt.yticks(fontsize=12)

plt.legend()
plt.show()

print(mpld3.fig_to_html(plt, d3_url=None, mpld3_url=None, no_extras=False, template_type='general', figid=None, use_http=False))


