import requests
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt,mpld3
from datetime import date


print("Welcome to python reporting")

session = requests.Session()
session.trust_env = False  # proxy ausschalten

pd.set_option('display.max_colwidth', None)  # die komplette Spalte anzeigen
pd.set_option('display.max_columns', None)  # alle Spalten anzeigen
# pd.set_option('display.max_rows', None)

df = pd.read_csv('identitaeten.txt', sep='\t')

# columns setzen
df.columns = ['jahrgang', 'geschlecht', 'count']

# unbekannte wegmachen
df = df[df['geschlecht'] != 3]
print(df.shape)

# code mit männlich, weiblich ersetzen
df.loc[df.geschlecht == 1, "geschlecht"] = "männlich"
df.loc[df.geschlecht == 2, "geschlecht"] = "weiblich"

df.sort_values('jahrgang', inplace=True, ascending=False)


df['jahrgang'] = df.apply(lambda row: int(date.today().strftime("%Y")) - int(row['jahrgang']), axis=1)



df['count'] = df.apply(lambda row: row['count'] * -1 if row['geschlecht'] == 'weiblich' else row['count'], axis=1)


plt.figure(figsize=(13, 10), dpi=80)
group_col = 'geschlecht'
order_of_bars = df['jahrgang'].unique()[::-1]
colors = [plt.cm.Spectral(i / float(len(df[group_col].unique()) - 1)) for i in range(len(df[group_col].unique()))]


for c, group in zip(colors, df[group_col].unique()):
    sns.barplot(x='count', y='jahrgang', orient='horizontal', dodge=False, data=df.loc[df[group_col] == group, :], order=order_of_bars, color=c, label=group)

plt.xlabel("Anzahl", )
plt.ylabel("Alter")
plt.yticks(fontsize=12)
plt.setp(plt.axes().get_yticklabels()[::2], visible=False)  # nur jeden zweiten Jahrgang anzeigen
plt.title("Pyramide", fontsize=22)
plt.legend()
plt.show()

print(mpld3.fig_to_html(plt, d3_url=None, mpld3_url=None, no_extras=False, template_type='general', figid=None, use_http=False))

# print(df)
#
# counts = [(df[df['geschlecht'] == 'männlich']['count']), (df[df['geschlecht'] == 'weiblich']['count'])]
#
# geschlechter = ['maennlich', 'weiblich']
#
# pyramid_chart = pygal.Pyramid(human_readable=True, legend_at_bottom=True)
# pyramid_chart.title = 'England population by age in 2010 (source: ons.gov.uk)'
# pyramid_chart.x_labels = map(lambda x: str(x) if not x % 5 else '', range(90))
#
#
# for geschlecht,count in zip(geschlechter, counts):
#     pyramid_chart.add(geschlecht, count)
# pyramid_chart.render_in_browser()
#
