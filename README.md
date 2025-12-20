# Probabilistic Outcome Modeling for Golf Head-to-head Betting Markets
*A Sports Betting Model*

# Goal and Outcome Summary
### What is a 3-ball golf bet?
It is a golf bet where bettors must pick the winner out of three preset golfers that are usually somewhere around the same skill level. These players often also play with eachother during that specific round of the tournament

### Why a 3-ball bet instead of tournament winners/ losers/ make the cut?
Our hypothesis to this analysis is that it will be easier to pick a singular winner out of a group of 3 golfers, rather then picking a singular golfer to place well in tournaments that can have a field of over 150 golfers. While this bet might be the *less sexy* compared to who will be lifting the trophy, we believe this kind of bet has untapped value that can be taken advantage of.

### What is our goal?
Our approach uses player performance statistics and weather data from a weather API into a predictive model for 3-ball outcomes. By combining historical performance data with weather while excluding market perception, we hope to produce more accurate probability than just the betting lines themselves. This work builds on existing research in sports analytics and betting prediction, which has generally focused on team sports or outcomes. We hope to find value within the betting lines to contribute to a method that can better inform others and highlight inefficiencies in the current market we see today. Because of the kind of growth of the sports betting world seen in todays world, we think it is even more important to innovate and take advantage of the opportunities becoming available to us.

# The Data
The dataset is made using a combination of the Datagolf archieve API (datagolf.com) as well as the Visual Weather Crossing API (visualcrossing.com)

The dataset used includes every PGA golfer's individual stats in every PGA golf tournament played since June 2nd, 2019 that offered betting odds (only excludes 2 tournaments). This means we will be able to analyze over 200 different golf tournaments, and tens of thousands of 3-Ball bets.