
# ================================================
# Python for Empirical Finance WS 2019/20
# Problem Set 1, Week 1
# Returns
# ================================================
#
# Prepared by Elmar Jakobs & Simon Walther
#

# In this problem set, you will get to know some real financial data,
# gain first experience with asset returns and perform a small comparison
# of long-term stock vs. bond investments.
#
# Enjoy!


# Setup
# -----
#
# Import packages for econometric analysis
# We'll need numpy, pandas, and matplotlib.pyplot
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# IMPORTANT !!!
#
# For submission, please change the status to 'SOLN', leave it empty while you are working on it.
# Once you set the status to 'SOLN', the code will not work on your machine any more. Don't worry,
# if it worked before setting status = 'SOLN', you're fine.
# If you hand in a file, with status not equal to 'SOLN', it will not count as a regular submission,
# which may lead to an evaluation with 0 points.
#
#status = ''
status = 'SOLN'  
# You can also just comment in this line


# Task 1: Data Management
# ----------------------------


# Read-in the provided csv file using the pandas read_csv function
# Hint: Use the 'parse_dates' attribute
#
prices = pd.read_csv('price_data.csv')
prices.head()

# Each column is labeled with the Ticker symbol of another asset. Here's the translation:
#   VBTIX - Vanguard Total Bond Market Index Fund, a broad bond index ETF
#   GE    - Stock of General Electric
#   WMT   - Stock of Walmart
#   IBM   - Stock of IBM
#   SPX   - S&P 500 index, a broad US equity index

# Read in the provided csv file 'tbill.csv' using the pandas read_csv function
#
riskfree = pd.read_csv('tbill.csv')
riskfree.head()





# Merge the price and riskfree rate data into a single data set based on their date
# Hint: Do the 'Date' columns of both DataFrames have the same data type? Check with the .info() function
# of the DataFrames
#
observations = pd.merge(prices,riskfree)
observations.head()


# Rename the 'TBILL' column into 'riskfree_rate'
#
observations = observations.rename(columns={'TBILL':'riskfree_rate'})
observations.head()

# Plot the evolution of the S&P 500 (= 'SPX') over time


plt.plot(observations['Date'],observations['SPX'])
plt.show()



# Check of intermediate result (1):
#
# HINT: Check for yourself: observations.loc[0, 'riskfree_rate'] should be 5.17
# observations.loc[0, 'riskfree_rate']
if status == 'SOLN':
    Test.assertEquals(str(np.round(observations.loc[5, 'riskfree_rate'],2))[0:4], str(checker_results.loc[1, 'result']), 'Check 1 failed')




# ============================================================================================


# Task 2: Asset Returns
# --------------------------

# Create a pandas DataFrame for returns. For the start, it'll only consist of the 'Date' column of the
# observations DataFrame
#
returns = pd.DataFrame()
returns['Date'] = observations['Date'].copy()
returns.head()

# Calculate log returns for all assets in the observations DataFrame, besides the 'riskfree_rate'.
# Store the results in the returns DataFrame.
# Hint: While it is not the only possible solution, the .shift() function of a pandas column might help you
# log returns benefits:
# 1. log-normality
# 2. approximate raw-log equality
# 3. time-additivity: compound return over periods is merely the difference in log between initial and final periods

returns['VBTIX'] = np.log(observations.VBTIX)-np.log(observations.VBTIX.shift())
returns['SPX'] = np.log(observations.SPX)-np.log(observations.SPX.shift())
returns['GE'] = np.log(observations.GE)-np.log(observations.GE.shift())
returns['WMT'] = np.log(observations.WMT)-np.log(observations.WMT.shift())
returns['IBM'] = np.log(observations.IBM)-np.log(observations.IBM.shift())

# Plot the return time series
# Q: Which charts look more similar to each other? Return Charts or Price Charts? Why?
# Return Charts. Calcualting log returns makes it possible to measure all variables in a comparable metric.


plt.plot(returns['Date'],returns['SPX'])
plt.show()



# Check of intermediate result (2):
#
# HINT: Check for yourself: returns.loc[1, 'GE'] should be -0.031175
# returns.loc[1, 'GE']
if status == 'SOLN':
    Test.assertEquals(str(np.round(returns.loc[5, 'IBM'],4))[0:7], str(checker_results.loc[2, 'result']), 'Check 2 failed')





# Calculate the daily mean and standard deviation (= volatility) of the return series
# Hint: We're looking for the population standard deviation
#
returns_stats_daily = pd.DataFrame(index = ['VBTIX', 'GE', 'WMT', 'IBM', 'SPX'], columns = ['mean', 'std'])
returns_stats_daily.loc['VBTIX', 'mean'] = returns['VBTIX'].mean()

returns_stats_daily.loc['GE', 'mean'] = returns['GE'].mean()
returns_stats_daily.loc['WMT', 'mean'] = returns['WMT'].mean()
returns_stats_daily.loc['IBM', 'mean'] = returns['IBM'].mean()
returns_stats_daily.loc['SPX', 'mean'] = returns['SPX'].mean()

returns_stats_daily.loc['VBTIX', 'std'] = returns['VBTIX'].std()
returns_stats_daily.loc['GE', 'std'] = returns['GE'].std()
returns_stats_daily.loc['WMT', 'std'] = returns['WMT'].std()
returns_stats_daily.loc['IBM', 'std'] = returns['IBM'].std()
returns_stats_daily.loc['SPX', 'std'] = returns['SPX'].std()

returns_stats_daily

# Check of intermediate result (3):
#
# HINT: Check for yourself: returns_stats_daily.loc['VBTIX', 'mean'] should be 0.0000352
# HINT: Check for yourself: returns_stats_daily.loc['VBTIX', 'std'] should be 0.002799
if status == 'SOLN':
    Test.assertEquals(str(np.round(returns_stats_daily.loc['SPX', 'mean'],5))[0:8], str(checker_results.loc[3, 'result']), 'Check 3 failed')
    Test.assertEquals(str(np.round(returns_stats_daily.loc['SPX', 'std'],4))[0:7], str(checker_results.loc[4, 'result']), 'Check 4 failed')




# Annualize the previously calculated mean and standard deviation
# Hint: There are 252 trading days in a year. The mean scales linearly, volatility scales with the square root
#       of trading days.
# Q: Why do we annualize?
# Usually, the return in given in annual terms. Since daily change is somehow too small to compare with each other.
returns_stats_annual = returns_stats_daily.copy()
returns_stats_annual.loc[:, 'mean'] = returns_stats_annual.loc[:, 'mean'] * 252

returns_stats_annual.loc[:, 'std'] = returns_stats_annual.loc[:, 'std'] * np.sqrt(252)

returns_stats_annual
# Q: What differences do you see in the mean log return and volatility(std)?
# Volatility has an obvious larger value than log return.
returns_stats_annual * 100

# Check of intermediate result (4):
#
# HINT: Check for yourself: returns_stats_annual.loc['VBTIX', 'mean'] should be 0.008872
# HINT: Check for yourself: returns_stats_annual.loc['VBTIX', 'std'] should be 0.04444
if status == 'SOLN':
    Test.assertEquals(str(np.round(returns_stats_annual.loc['SPX', 'mean'],3))[0:6], str(checker_results.loc[5, 'result']), 'Check 5 failed')
    Test.assertEquals(str(np.round(returns_stats_annual.loc['SPX', 'std'],3))[0:6], str(checker_results.loc[6, 'result']), 'Check 6 failed')
    


# Calculate the correlation between 'VBTIX', 'SPX' and 'GE'
#  Hint: Use the '.corr' function for data frames
#
returns_corr = returns[['VBTIX', 'SPX','GE']].corr()
returns_corr
print(np.round(returns_corr, 2))
# Q: What can you tell about the correlation between the bond and equity index? What about GE and the equity index?
# Bond Index(VBTIX) and Equity Index(SPX) are negatively correlated.
# GE and Equity Index are positively correlated since GE is a part of SPX. 


# Task 3: Sharpe Ratio
# ---------------------------------------

# The Sharpe ratio is defined as
# SR = (E(r) - r_f) / std(r).
# It's the expected return above the risk free rate (risk premium) divided by its volatility.
# It tells us how much premium we get for a unit of risk.
# Here, we'll use its version with log returns
#

# The riskfree rates in the observations DataFrame is already the log return of a risk free bond.
# By convention, the return is given in annual terms. For the other assets, we have daily log returns.
# Also, they are given in percentage terms, while our other returns are in actual numbers (i.e. 0.01 instead of 1%)
# Hence, scale the riskfree_rate to the rate of a single day in actual numbers (there are 252 trading days in a year).
#


observations['riskfree_rate'] = observations['riskfree_rate']/(100*252)
# daily riskfree rate
observations

# Calculate daily excess returns for all 5 assets
#
excess_returns = returns.copy()

excess_returns['VBTIX'] = returns['VBTIX'] -observations['riskfree_rate']
excess_returns['SPX'] = returns['SPX'] -observations['riskfree_rate']
excess_returns['GE'] = returns['GE'] -observations['riskfree_rate']
excess_returns['WMT'] = returns['WMT'] -observations['riskfree_rate']
excess_returns['IBM'] = returns['IBM'] -observations['riskfree_rate']

excess_returns.tail()



# Calculate daily mean and volatility of excess returns
#
excess_stats_daily = pd.DataFrame(index = ['VBTIX', 'GE', 'WMT', 'IBM', 'SPX'], columns = ['mean', 'std'])
excess_stats_daily['mean'] = excess_returns.mean()
excess_stats_daily['std'] = excess_returns.std()
excess_stats_daily


# Calculate the daily Sharpe ratio of the 5 assets
#
SR_daily = excess_stats_daily['mean']/excess_stats_daily['std']
SR_daily
# Q: Are there differences between the stock, stock index, and bond Sharpe ratio?
# GE and IBM with negative sharp ratio
# SPX has the highest sharp ratio

# Check of intermediate result (5):
#
# HINT: Check for yourself: SR_daily['GE'] should be -0.02039
# SR_daily['GE']
if status == 'SOLN':
    Test.assertEquals(str(np.round(SR_daily['SPX'],3))[0:6], str(checker_results.loc[7, 'result']), 'Check 7 failed')
    Test.assertEquals(str(np.round(SR_daily['VBTIX'],4))[0:7], str(checker_results.loc[8, 'result']), 'Check 8 failed')
    
         
# Calculate the annual Sharpe ratio of the 5 assets
# Hint: Annualize the excess mean return and volatility first
#
SR_annual = (excess_stats_daily['mean']*252) / (excess_stats_daily['std']* np.sqrt(252))
SR_annual

# Calculate the 10-year Sharpe ratio of the 5 assets
# Hint: Just assume all years have 252 trading days.
#
SR_10y = (excess_stats_daily['mean']*252*10) / (excess_stats_daily['std']* np.sqrt(252*10))
SR_10y
# Q: Any differences to the daily horizon?
# Sharp Ration is larger as the time scale expands.
# Q: As a long-term investor (say, 10 years and more) what would you invest in? The stock index (SPX) or bond index (VBTIX)?
# As long-term investment, SPX would be the best choice.


# Check of intermediate result (6):
#
# HINT: Check for yourself: SR_10y['GE'] should be -1.0234
if status == 'SOLN':
    Test.assertEquals(str(np.round(SR_10y['SPX'],3))[0:6], str(checker_results.loc[9, 'result']), 'Check 9 failed')



# Task 4: Long-run equity investments
# ---------------------------------------------------------

# Looks like investing in stocks is very profitable in the long run.
# So let's put all our money in stocks! We're young!  ;)
# Not so fast.

# Assume for a moment that log returns are normally distributed. Of course, they're not, but for the first
# problem set we can be a bit inaccurate.
# The normal distribution is defined by their mean and standard deviation that we just pinned down in the previous tasks.

from scipy.stats import norm   # Gives access to functionality about normal distributions

# What is the probability of a 30% loss after 1 year of holding the SPX?
# Hint: Use the mean and volatility of log returns (not excess returns).
# Hint: Use the function norm.cdf(...) to get the probability of a certain return, given its mean and volatility.
#       If you're unsure how to use that function, google it.
# Hint: A 30% loss translates to a gross return of 0.7, which needs still to be translated into a log return.
#
prob_1y_spx = norm.cdf(returns_stats_annual.loc['SPX','mean']*0.7,returns_stats_annual.loc['SPX','mean'],returns_stats_annual.loc['SPX','std']) 
prob_1y_spx

# What is the probability of a 30% loss after 10 years of holding the SPX?
prob_10y_spx = norm.cdf(returns_stats_annual.loc['SPX','mean']*0.7*10,returns_stats_annual.loc['SPX','mean']*10,returns_stats_annual.loc['SPX','std']*np.sqrt(10) 
prob_10y_spx

# What is the probability of a 30% loss after 1 year of holding the VBTIX?
#
prob_1y_vbtix = norm.cdf(returns_stats_annual.loc['VBTIX','mean']*0.7,returns_stats_annual.loc['VBTIX','mean'],returns_stats_annual.loc['VBTIX','std']) 
prob_1y_vbtix

# What is the probability of a 30% loss after 10 years of holding the VBTIX?
prob_10y_vbtix = norm.cdf(returns_stats_annual.loc['VBTIX','mean']*0.7*10,returns_stats_annual.loc['VBTIX','mean']*10,returns_stats_annual.loc['VBTIX','std']*np.sqrt(10) 
prob_10y_vbtix

# Q: What do you think about the equity investment now? Would you prefer the bond investment?
# The equity investment has higher risk. Whether invest or not, it depends on the individual risk preference.                     

# Check of intermediate result (7):
#
if status == 'SOLN':
    Test.assertEquals(str(np.round(prob_10y_spx,3))[0:5], str(checker_results.loc[10, 'result']), 'Check 10 failed')

# ----------------------- END OF PROBLEM SET 1 ---------------------------------------------

