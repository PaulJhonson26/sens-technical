from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)

sc_films = pd.read_csv('sc_films.csv')
top_100_films = pd.read_csv('top_100_films.csv')

top_with_stats = pd.DataFrame()

for i in range(100):
        top100_row = top_100_films.loc[i]
        top100_id = top100_row['id']
        row_with_stats = sc_films[sc_films['id'] == top100_id]
        top_with_stats = top_with_stats._append(row_with_stats, ignore_index = True)

top_indeps = top_with_stats.iloc[:, 2:17]

top_indeps["date_release"] = top_indeps["date_release"].str[:4].fillna(2000).astype(int)-2000
sc_indeps = sc_films.iloc[:, 2:17]
sc_indeps["date_release"] = sc_indeps["date_release"].str[:4].fillna(2000).astype(int)-2000

X_train, X_test, Y_train, Y_test = train_test_split(top_indeps[["date_release", 'rating', 'wish_list_count', 'review_count', 'list_count', 'n_1', 'n_2', 'n_3', 'n_4', 'n_5', 'n_6', 'n_7', 'n_8', 'n_9', 'n_10']]
, top_100_films["rank"], test_size=0.1, random_state=26)

# Lasso best alpha = 2090
lasso_model = Lasso(alpha=2090)
lasso_model.fit(X_train, Y_train)


lasso_predictions = lasso_model.predict(X_test)
lasso_mse = mean_squared_error(Y_test, lasso_predictions)
print("Lasso Mean Squared Error: {lasso_mse}")

pred = lasso_model.predict(sc_indeps)
print(pred)
print(lasso_model.coef_)

#Normal Linear Regresssion
#model = LinearRegression()

# Fit the model to your data
#model.fit(X_train, Y_train)

#pred = model.predict(sc_indeps)
# print(sc_indeps)

sc_films_ranked = sc_films
sc_films_ranked["rating_count"] = pred

sc_films_ranked = sc_films_ranked.sort_values(by='rating_count', ascending=True)
sc_films_ranked.reset_index(drop=True, inplace=True)

def finalize(sc_films_ranked):
    final_frame = pd.DataFrame(columns=['rank', 'id', 'title', 'rating_count'])
    for i in range(200):
        final_frame = final_frame._append(sc_films_ranked.iloc[i][["id", "title", "rating_count"]], ignore_index=True)
    final_frame["rank"] = range(1, 201)
    return final_frame


def print_test_answer(solution):
    score = 0
    for i in range(0, 100):
        top100_row = top_100_films.loc[i]
        top100_id = top100_row['id']
        myrank = solution[solution['id'] == top100_id]
        myrank_number = myrank.index[0]

        print("I ranked it: ", myrank_number, "it should be: ", i)

        score += abs(i - myrank_number)

    print("predictions are off by: ",score/100 , " on average")

print_test_answer(sc_films_ranked)
final_frame = finalize(sc_films_ranked)
final_frame.to_csv('top_200_films_linear.csv', index=False)
