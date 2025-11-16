Inside all_data, we have the following csv data
1. exreturns -> returns data from Jan/1990 - Dec/2019
2. options  -> options data from Jan/1996 - Dec/2019
3. returns  -> returns data from Jan/2000 - Dec/2019
4. RV5  -> realized variance data from Jan/2000 - Dec/2019

Then we also have 4 noise data generated from the standard Gaussian distribution
each of shape [10000, 250]. They are:
1. samples_1
2. samples_2
3. samples_3
4. samples_4