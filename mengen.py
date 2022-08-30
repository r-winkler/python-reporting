import requests
import pandas as pd
import pygal
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

print("Welcome to python reporting")

session = requests.Session()
session.trust_env = False  # proxy ausschalten

pd.set_option('display.max_colwidth', None)  # die komplette Spalte anzeigen
pd.set_option('display.max_columns', None)  # alle Spalten anzeigen
# pd.set_option('display.max_rows', None)

gauge = pygal.SolidGauge(inner_radius=0.70)
gauge.add('Dossier', [{'value': 839648, 'max_value': 2230828}])
gauge.add('Identität', [{'value': 1741076, 'max_value': 2230828}])
gauge.add('Beabeitungsvermerk', [{'value': 7991, 'max_value': 2230828}])
gauge.add('Strafverfahren', [{'value': 105813, 'max_value': 2230828}])
gauge.add('Grundurteile', [{'value': 1144811, 'max_value': 2230828}])
gauge.add('Nen', [{'value': 252843, 'max_value': 2230828}])
gauge.add('Delikte', [{'value': 2173468, 'max_value': 2230828}])
gauge.add('Sanktionen', [{'value': 2230828, 'max_value': 2230828}])
gauge.add('Strafzusatz', [{'value': 85316, 'max_value': 2230828}])
gauge.add('Pcn', [{'value': 3295, 'max_value': 2230828}])
gauge.add('Strafzumessungsgrund', [{'value': 291011, 'max_value': 2230828}])
gauge.add('Ruhezeiten', [{'value': 122, 'max_value': 2230828}])
gauge.add('Landesverweise', [{'value': 3418, 'max_value': 2230828}])

gauge.render_in_browser()


