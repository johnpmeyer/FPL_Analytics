import pandas as pd
import cvxpy as cp
import numpy as np

#load data
data_full = pd.read_csv("epl_optimization_data.csv")
data_starters = pd.read_csv("final_optimized_team_alt.csv")

#get indices of selected players from starting XI in 'fpl_optimization'
relevant_idx = np.array(data_starters[['Unnamed: 0']]).flatten()

#drop those players
data_full.drop(relevant_idx, axis=0, inplace=True)
data_adj = data_full.drop(columns=["Unnamed: 0", "team_name", "position"])

#data prep for numpy
data_np = np.matrix(data_adj)

#getting variables setup in numpy
x = np.reshape(data_np[:, 3], (514, 1)) #total points
p = np.reshape(data_np[:, 6], (514, 1)) #total cost

#goalkeeper, defender, midfielder, and forward
a = np.reshape(data_np[:, 29], (514, 1)) #goalkeeper
b = np.reshape(data_np[:, 27], (514, 1)) #defender
d = np.reshape(data_np[:, 30], (514, 1)) #midfielder
e = np.reshape(data_np[:, 28], (514, 1)) #total points

#decision variable -- invest in player i or not.
c = cp.Variable(514, boolean=True)


#objective function
objective = cp.Maximize(c @ x) #maximize points

#constraints, noting that this is for the bench. These are hard-coded right now based on results from fpl_optimization.py
constraints = [
    c @ a == 1, #one keeper
    c @ b == 0, #0 defenders
    c @ d == 1, #one midfielder
    c @ e == 2, #two forwards
    c @ p <= 200 #80M cost or less.
]
 #constraints so each team only has at most three representatives
for i in range(7, 27):
    team_data = np.reshape(data_np[:, i], (514, 1))
    constraints.append(c @ team_data <= 3)

model = cp.Problem(objective, constraints)
model.solve()

print(model.value)
print(c.value @ p)

c_final = c.value
c_idx = np.where(c_final==1)[0]

final_team_df = data_full.iloc[c_idx, :]
final_team_df.drop(columns=["Unnamed: 0"], inplace=True)

final_team_df.to_csv("final_optimized_bench_alt.csv")