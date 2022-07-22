import pandas as pd
import cvxpy as cp
import numpy as np

#load data
data = pd.read_csv("epl_optimization_data.csv")
data_adj = data.drop(columns=["Unnamed: 0", "team_name", "position"])

#data prep for numpy
data_np = np.matrix(data_adj)

#getting variables setup in numpy
x = np.reshape(data_np[:, 3], (525, 1)) #total points
p = np.reshape(data_np[:, 6], (525, 1)) #total cost

#goalkeeper, defender, midfielder, and forward
a = np.reshape(data_np[:, 29], (525, 1)) #goalkeeper
b = np.reshape(data_np[:, 27], (525, 1)) #defender
d = np.reshape(data_np[:, 30], (525, 1)) #midfielder
e = np.reshape(data_np[:, 28], (525, 1)) #total points

#decision variable -- invest in player i or not.
c = cp.Variable(525, boolean=True)


#objective function
objective = cp.Maximize(c @ x) #maximize points

#constraints, noting that this is just for a starting XI.
constraints = [
    c @ a == 1, #one keeper
    c @ b <= 5, #five or less defenders
    c @ d <= 5, #five or less midfielders
    c @ e <= 3, #three oe less forwards
    c @ a + c @ b + c @ d + c @ e == 11, #eleven players total
    c @ b >= 3, #three or more defenders
    c @ d >= 3, #three or more midfielders
    c @ e >= 1, #one or more forwards
    c @ p <= 800 #80M cost or less.
]
 #constraints so each team only has at most three representatives
for i in range(7, 27):
    team_data = np.reshape(data_np[:, i], (525, 1))
    constraints.append(c @ team_data <= 3)

model = cp.Problem(objective, constraints)
model.solve()

print(model.value)
print(c.value @ p)

c_final = c.value
c_idx = np.where(c_final==1)[0]

final_team_df = data.iloc[c_idx, :]
final_team_df.drop(columns=["Unnamed: 0"], inplace=True)

final_team_df.to_csv("final_optimized_team_alt.csv")