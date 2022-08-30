import pandas as pd
import datetime
import random


pd.set_option('display.max_colwidth', None)  # die komplette Spalte anzeigen
pd.set_option('display.max_columns', None)  # alle Spalten anzeigen
# pd.set_option('display.max_rows', None)

sample_size = 8000000

nachnamen = pd.read_csv('datasets/nachnamen_proplz.csv')
nachnamen = nachnamen[nachnamen['Nachname'].notnull()]
nachnamen = nachnamen['Nachname'].drop_duplicates()
nachnamen = nachnamen.str.upper()


vornamen = pd.read_excel('datasets/vornamen_bfs.xlsx', skiprows=20, skipfooter=20)
vornamen.columns =['Vorname', 'w', 'm']
vornamen['Vorname'] = vornamen['Vorname'].str.upper()



start_date = datetime.date(1965, 1, 1)
end_date = datetime.date(1995, 1, 1)

time_between_dates = end_date - start_date
days_between_dates = time_between_dates.days

year = []
month = []
day = []
initialsuche_relevant = []
for i in range(0, sample_size):
    random_number_of_days = random.randrange(days_between_dates)
    n = start_date + datetime.timedelta(days=random_number_of_days)
    year.append(n.year)
    month.append(n.month)
    day.append(n.day)
    rel = random.randrange(1,4,1)
    if rel == 1:
        initialsuche_relevant.append('true')
    else:
        initialsuche_relevant.append('false')

dates = pd.DataFrame(list(zip(year, month, day, initialsuche_relevant)), columns=['year', 'month', 'day', 'initialsuche_relevant'])


tuples = list(zip(nachnamen.sample(sample_size, replace=True), vornamen.sample(sample_size, replace=True)['Vorname'], dates['year'], dates['month'], dates['day'], initialsuche_relevant))
df = pd.DataFrame(tuples, columns = ['Nachname', 'Vorname', 'Jahr', 'Monat', 'Tag', 'Initialsuche_Relevant'])


df.to_csv('datasets/names_of_switzerland.csv', header=False, index=False)

exit()
