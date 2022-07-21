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
                                   'total_points', 'points_per_game', 'minutes',
                                    'cost', 'position', 'team'])

#extracting player info the good-ole fashioned way.
for item in players:
    row_data = \
        {'first_name': [item['first_name']],
         'second_name': [item['second_name']],
         'web_name': [item['web_name']],
         'total_points': [item['total_points']],
         'points_per_game': [item['points_per_game']],
         'minutes': [item['minutes']],
         'cost': [item['now_cost']],
         'position': [item['element_type']],
         'team': [item['team']],
        }
    row_df = pd.DataFrame(row_data)
    player_df = pd.concat([player_df, row_df])

#extracting teams data using pd.normalize()
teams = pd.json_normalize(r['teams'])
teams = teams[['id', 'name']]

#positions
positions = pd.json_normalize(r['element_types'])
positions = positions[["id", "singular_name_short"]]

#joining data.
#joining teams
joined_data = player_df.merge(teams, how="left", left_on = "team", right_on = "id")
joined_data = joined_data.drop(columns=['team', 'id'])
joined_data.rename(columns = {'name': 'team_name'}, inplace = True)

#joining positions
joined_data = joined_data.merge(positions, how="left", left_on = "position", right_on = "id")
joined_data = joined_data.drop(columns=['position', 'id'])
joined_data.rename(columns = {'singular_name_short': 'position'}, inplace = True)

#creating dummies to make subsequent optimization problem easier
dummies_team = joined_data[['team_name']]
dummies_team = pd.get_dummies(dummies_team)

#doing similar for the positions
dummies_pos = joined_data[['position']]
dummies_pos = pd.get_dummies(dummies_pos)

#final data
final_data = pd.concat([joined_data, dummies_team], axis=1)
final_data = pd.concat([final_data, dummies_pos], axis = 1)
final_data = final_data.drop(columns = ['team_name', 'position'])

final_data.to_csv("epl_optimization_data.csv")

