import pandas as pd
import random
import copy
import matplotlib.pyplot as plt
import numpy as np

#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)

sc_films = pd.read_csv('sc_films.csv')
top_100_films = pd.read_csv('top_100_films.csv')

#method sorts a dataframe by applying weights to each attribute, calculating a score and sorting based on that score
def sort(best_solution, s_w):

    for i in range(2780): #for every row in the dataframe
        current_row = best_solution.loc[i]
        score = 0
        for j in range(15):# apply each weight to each column
            if j == 0:# converting year to int
                if type(current_row[j+2]) is not float:#some years aren't registered so ignoring those
                    score += s_w[j]*(int(current_row[j+2][:4])-2015) #multiply weight by year (-2015 is to standardize)
            else:
                    score += s_w[j]*current_row[j+2]

        best_solution.loc[i, "rating_count"] = score #adding score column to dataframe
    best_solution = best_solution.sort_values(by='rating_count', ascending=False) #sorting by score
    best_solution.reset_index(drop=True, inplace=True) #reset index to new sorted order
    return best_solution

#method compares the rank of top 100 movies against our current solution
def test_answer(solution):
    score = 0
    for i in range(0, 99): #for every row in top 100 movie
        top100_row = top_100_films.loc[i]
        top100_id = top100_row['id'] #find its id and compare corresponding id in solution
        myrank = solution[solution['id'] == top100_id]
        myrank_number = myrank.index[0]
        score += abs(i - myrank_number) #calculate score based on absolute difference in position
    return score

#same method as above but for displaying final results
def print_test_answer(solution):
    score = 0
    for i in range(100):
        top100_row = top_100_films.loc[i]
        top100_id = top100_row['id']
        myrank = solution[solution['id'] == top100_id]


        if not myrank.empty:
            myrank_number = myrank.index[0]
            print("I ranked it: ", myrank_number, "it should be: ", i)
            score += abs(i - myrank_number)

        else:
            print("No match found for top100_id:", top100_id)
            score += 200
    print("predictions are off by: ",score/100 , " on average")

#initialize start of algorithm
solution = sc_films
solution["rating_count"] = 0
best_score = float('inf')
best_solution = solution
#s_w is the weights associated to each column in the dataframe (excluding id and title which should have no impact)
#s_w = [1.4494125133328308, 3.447909116110372, 2.5058525699837486, -1.4835631120411323, 0.5231162644731713, -0.49223891309495027, 0.3019758134820161, 1.5479072196041235, -1.270644498999873, -1.39576703053954, 0.5829263854314388, 0.6525957825103781, -0.27451243334954856, -0.13131296805104342, 5.750111650709516]# 500 its, score: 1954
s_w = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
best_sw = copy.deepcopy(s_w)
score_evolution = [] # arrays to keep track of score changes over time to see how it trends
best_score_evolution = []
#number of iterations can change but I found that 500 is very sufficient as solution plateus around 500
for i in range(500):
    randIdx = random.randint(0, 14)
    randDelt = random.uniform(-1,1)
    s_w[randIdx] += randDelt #randomly change one of the 15 weights by an amount between -1 and 1
    solution = sort(best_solution, s_w) #sort our current solution based on these new weights
    current_score = test_answer(solution) #calculate the score of our sorted solution
    print(i," - final score: ", current_score)
    #keeping track of score progression
    if current_score <= 15000:
        score_evolution.append(current_score)
    # update best solution weights and score if a better solution is found
    if (current_score < best_score):
        best_score_evolution.append(current_score)
        best_solution = solution
        best_score = current_score
        best_sw = copy.deepcopy(s_w)
    else:
        best_score_evolution.append(best_score)
        s_w = copy.deepcopy(best_sw)

# converting solution into top_200 format
def finalize(best_solution):
    final_frame = pd.DataFrame(columns=['rank', 'id', 'title', 'rating_count'])
    for i in range(200):
        final_frame = final_frame._append(best_solution.iloc[i][["id", "title", "rating_count"]], ignore_index=True)
    final_frame["rank"] = range(1, 201)
    return final_frame

final_frame = finalize(best_solution)
final_frame.to_csv('top_200_films.csv', index=False)
data = np.array(score_evolution)
data2 = np.array(best_score_evolution)

# Create the first line plot
# plt.figure()
# plt.plot(range(len(data)), data)
# plt.xlabel('Index')
# plt.ylabel('Score')
# plt.title('Score Evolution')
# plt.savefig('score_evolution.png', format='png')
# plt.close()

# # Create the second line plot
# plt.figure()
# plt.plot(range(len(data2)), data2)
# plt.xlabel('Index')
# plt.ylabel('Score')
# plt.title('Best Score Evolution')
# plt.savefig('best_score_evolution.png', format='png')
# plt.close()

print("best weights: ", best_sw)
print("best_score: ", best_score)
print("best_solution: ", best_solution)
print_test_answer(best_solution)
