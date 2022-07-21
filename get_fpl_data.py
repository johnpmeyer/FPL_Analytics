import requests, json
from pprint import pprint

import pandas as pd

# base url for all FPL API endpoints
base_url = 'https://fantasy.premierleague.com/api/'

# get data from bootstrap-static endpoint
r = requests.get(base_url+'bootstrap-static/').json()

# show the top level fields
pprint(r, indent=2, depth=1, compact=True)

#create players dictionary from master JSON
players = r['elements']

player_df = pd.DataFrame(columns = ['first_name', 'second_name', 'web_name',
                                   'total_points', 'points_per_game', 'minutes', 'team'])

#extracting player info the good-ole fashioned way.
for item in players:
    row_data = \
        {'first_name': [item['first_name']],
         'second_name': [item['second_name']],
         'web_name': [item['web_name']],
         'total_points': [item['total_points']],
         'points_per_game': [item['points_per_game']],
         'minutes': [item['minutes']],
         'team': [item['team']],
        }
    row_df = pd.DataFrame(row_data)
    player_df = pd.concat([player_df, row_df])

#extracting teams data using pd.normalize()
teams = pd.json_normalize(r['teams'])
teams = teams[['id', 'name']]

#joining data.
joined_data = player_df.merge(teams, how="left", left_on = "team", right_on = "id")
joined_data = joined_data.drop(columns=['team', 'id'])
joined_data.rename(columns = {'name': 'team_name'}, inplace = True)

#creating dummies to make subsequent optimization problem easier
dummies = joined_data[['team_name']]
dummies = pd.get_dummies(dummies)

#final data
final_data = pd.concat([joined_data, dummies], axis=1)

final_data.to_csv("epl_optimization_data.csv")

