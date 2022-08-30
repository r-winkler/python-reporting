import requests
import pandas as pd
import pygal
from datetime import date
import seaborn as sns
import matplotlib
# matplotlib.use('WebAgg')
import matplotlib.pyplot as plt,mpld3

print("Welcome to python reporting")

session = requests.Session()
session.trust_env = False  # proxy ausschalten

pd.set_option('display.max_colwidth', None)  # die komplette Spalte anzeigen
pd.set_option('display.max_columns', None)  # alle Spalten anzeigen
# pd.set_option('display.max_rows', None)


sanktionen = pd.DataFrame(session.get("https://eap0249:30014/rest/code/codetable/sanktion", verify=False).json())
tatbestande = pd.DataFrame(session.get("https://eap0249:30014/rest/code/codetable/tatbestand", verify=False).json())

data = {'code': [
    'B000227',
    'B002705',
    'B005163',
    'B003077',
    'B004092',
    'B005162',
    'B005236',
    'B001844',
    'B001379',
    'B000225'
], 'count': [
    157077,
    93338,
    90489,
    88951,
    68298,
    66415,
    64936,
    62793,
    61909,
    57905
]}

df = pd.DataFrame(data)

joined = pd.merge(df, tatbestande)  # automatic inner join on code
joined = joined.filter(['code', 'bezeichnungI18nDe', 'count'])

joined.to_csv(r'/home/isc-wre/Downloads/report.txt', sep=' ', index=False)

line_chart = pygal.Pie()
line_chart.title = 'Top Delikte'
for index, row in joined.iterrows():
    line_chart.add(row['bezeichnungI18nDe'], row['count'])

line_chart.render_to_file('/home/isc-wre/Downloads/report.svg')  # geht nicht, nur schwarzes png
line_chart.render_to_png('/home/isc-wre/Downloads/report.png')
line_chart.render_in_browser()

####################################################################################
####################################################################################
####################################################################################


df = pd.read_csv('delikte.txt', sep='\t')
df.columns = ['jahrgang', 'code', 'count']

df['jahrgang'] = df.apply(lambda row: int(date.today().strftime("%Y")) - int(row['jahrgang']), axis=1)

df['jahrgang'] = df['jahrgang'].apply(lambda x: 20 if x < 30 else x)
df['jahrgang'] = df['jahrgang'].apply(lambda x: 30 if x < 40 and x >= 30 else x)
df['jahrgang'] = df['jahrgang'].apply(lambda x: 40 if x < 50 and x >= 40 else x)
df['jahrgang'] = df['jahrgang'].apply(lambda x: 50 if x < 60 and x >= 50 else x)
df['jahrgang'] = df['jahrgang'].apply(lambda x: 60 if x < 70 and x >= 60 else x)
df['jahrgang'] = df['jahrgang'].apply(lambda x: 70 if x >= 70 else x)


df = df.groupby(['jahrgang', 'code']).sum('count').reset_index()


df = df.pivot(index='code', columns='jahrgang', values='count').reset_index()

df = df.fillna(0)


df = df[df['code'].isin(['B000227',
                        'B002705',
                        'B005163',
                        'B003077',
                        'B004092',
                        'B005162',
                        'B005236',
                        'B001844',
                        'B001379',
                        'B000225'])]

df['sum'] = df.apply(lambda row: row[1:].sum(), axis=1)

df.sort_values('sum', inplace=True, ascending=False)


sns.set(style="whitegrid")


# joined = pd.merge(df, tatbestande)  # automatic inner join on code
# joined = joined.filter(['code', 'jahrgang', 'bezeichnungI18nDe', 'count'])
# print(joined)

# Make the PairGrid
g = sns.PairGrid(df,
                 x_vars=[20, 30, 40,
                         50, 60, 70], y_vars=["code"],
                 height=10, aspect=.25)

# Draw a dot plot using the stripplot function
g.map(sns.stripplot, size=10, orient="h",
      palette="ch:s=1,r=-.1,h=1_r", linewidth=1, edgecolor="w")

# Use the same x axis limits on all columns and add better labels
g.set(xlim=(0, 45000), xlabel="Anzahl", ylabel="")


# Use semantically meaningful titles for the columns
titles = ["<30", "30-40", "40-50",
          "50-60", "60-70", "70+"]

for ax, title in zip(g.axes.flat, titles):

    # Set a different title for each axes
    ax.set(title=title)

    # Make the grid horizontal instead of vertical
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)

sns.despine(left=True, bottom=True)

g.savefig('./Downloads/top-delikte-nach-alter.png')

plt.show()
