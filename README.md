# sensCritique-technical

### Linear regression model
I started off with a simple multiple linear regression model from sci-kit-learn's libraries. with only 100 data points to create a model, I used Lasso Regularization to improve the result greatly.
This model was about 20 ranks off on average with respect to the top 100.

### Genetic-inspired model
I also had an idea of a sort of genetic algorithm model I could use. I played around with an algorithm free from any libraries.
It was not very efficient but it gave good results that similarly to the linear model leveled out at approximately 19 ranks off on average. This is shown in the two graphs plotted.

### Improvements?
I could have attempted using a neural network but with such little training data it seemed inadvisable.
There are also definitely better libraries to code a genetic algorithm but I'm not sure they would have much improvement on either model as they both leveled out around the same value. I also had fun coding my own algorithm :)

### Result
The top 200 CSV files are my final rankings. I kept both by curiosity but the top_200_films.csv is the main result.

