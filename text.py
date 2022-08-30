# coding = utf-8
import textdistance
import pandas as pd

pd.set_option('display.max_colwidth', None)  # die komplette Spalte anzeigen
pd.set_option('display.max_columns', None)  # alle Spalten anzeigen
pd.set_option('display.max_rows', None)

df = pd.read_csv('datasets/names_of_switzerland.csv')
df.columns = ['Nachname', 'Vorname']
indexNachname = 0
indexVorname = 1

thresholdNachname = 0.9
thresholdVorname = 0.85

# search = input("Enter search:\n")
searchNachname = "rosser"
searchVorname = "benedikt"

# case-insensitive search, vielleicht ist auch eine Accent-insensitve Suche sinnvoll (aber zu aufwändig wahrscheinlich)
searchNachname = searchNachname.title()  # first letter uppercase
searchVorname = searchVorname.title()

# es gibt max 6000 Einträge für S, weitere kristische sind B und M
# select geburtsdatum_jahr, count(*) from identitaet i where name like 'S%' group by i.geburtsdatum_jahr order by 2 desc;

# erster Buchstabe des Nachnamen muss übereinstimmen (case-insensitive)
df = df[df['Nachname'].str.startswith(searchNachname[0])]

df['Nachname_Similarity'] = df.apply(lambda row: textdistance.jaro_winkler.normalized_similarity(searchNachname, row.iloc[indexNachname]), axis=1)
df['Vorname_Similarity'] = df.apply(lambda row: textdistance.jaro_winkler.normalized_similarity(searchVorname, row.iloc[indexVorname]), axis=1)

df = df[(df['Nachname_Similarity'] > thresholdNachname) & (df['Vorname_Similarity'] > thresholdVorname)]

print(df.head(100))

exit()
