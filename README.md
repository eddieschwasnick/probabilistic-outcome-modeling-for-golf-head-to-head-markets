# Probabilistic Outcome Modeling for Golf Head-to-head Betting Markets
*Quantitative Sports Betting and Applied Machine Learning Research*

# Goal and Outcome Summary
### What is a 3-ball golf bet?
It is a golf bet where bettors must pick the winner out of three preset golfers that are usually somewhere around the same skill level. These players often also play with eachother during that specific round of the tournament

### Why a 3-ball bet instead of tournament winners/ losers/ make the cut?
Our hypothesis to this analysis is that it will be easier to pick a singular winner out of a group of 3 golfers, rather then picking a singular golfer to place well in tournaments that can have a field of over 150 golfers. Picking from a field this large can lead to added variability and uncertainty.

This structural reduction in variance makes 3-ball markets well-suited for probabilistic modeling, as relative player performance can be estimated more precisely than absolute tournament placement.

### What is the goal of this project?
While classification accuracy provides a baseline metric, the primary objective of this framework is probabilistic calibration rather than pure accuracy, as betting decisions depend on expected value rather than correct classification alone.

The player feautures are historical round-by-round performance metrics, while we also incorperated external player factors such as weather to capture any round specific effects of each match. It was also taken into account during training to not leak future data while predicting to ensure that predictions are driven solely by observable performance and contextual data. By estimating outcomes, this framework is designed to try to evaluate whether there are any inefficencies in the current market between statistical information and betting lines.

### Outcome Summary
Unknowing which model would fit our data the best, we implemented a Random Forest, Logistic Regression, and a Neural Network to find which would have the best fit.

Theoretically if every player had the same betting odds, you would need an accuracy over 33.3% to beat random guessing. But, in our dataset, we see a bias towards the first position (~46%), probably from the way it was recorded after the fact. Despite this change, because of sportsbook betting odds and fees, most professional algorithms typically need at least a 52-53% win rate in order to be profitable.

Random Forest is the best (as expected):
With that being said our Random Forest performed with roughly 55% accuracy, which greatly outperformed random guessing and is somewhere around the low end of an actual sports betting model. This was good, but not perfect, so it was a useful baseline for verifying that our data preprocessing and rolling average features were correct. 

The other models + some important notes:
After tweaking hyperparameters, Logistic Regression was about 50% accurate, and the Neural Network was around 41% accurate. Although Random Forest is the best of what we’ve tested, we think models like XGBoost or LightGBM could be even better. Along with that, we’d like to run a deeper analysis into comparing DraftKing's betting odds to ours to determine if our model is better than their estimations. 

Conclusion:
Overall, the potential of our work of integrating player, betting market, and weather data together to predict betting outcomes is clear. This also lays the groundwork to test future models using this cleaned data to potentially be even more accurate, or compare to online betting and spot the trends when market sentiment disagrees with statistical information.


# The Data
The dataset is made using a combination of the Datagolf archieve API (datagolf.com) as well as the Visual Weather Crossing API (visualcrossing.com)

The dataset used includes every PGA golfer's individual stats in every PGA golf tournament played since June 2nd, 2019 that offered betting odds (only excludes 2 tournaments). This means we will be able to analyze over 200 different golf tournaments, and tens of thousands of 3-Ball bets.

## IMPORTANT NOTES
Temporal Bias due to Data Availability.
The dataset begins in the 2019 PGA season, which introduces a limitation for player specific-historical features. There are many veteran golfers whos careers started years before 2019. Because of this the model does not observe their full performance histroy, and causes our models early on rolling averages to be very sparse or event undefined. As the sample of tournaments increases, and more betting rounds are observed, our rolling statistics become more and more informative and should lead to an increase in model performance over time.

