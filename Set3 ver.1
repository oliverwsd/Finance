#
# ================================================
# Empirical Finance WS 19/20
# Problem Set 3, Week 5
# Linear Factor Models and Equity Risk Premium
# ================================================
#
# Prepared by Elmar Jakobs & Simon Walther
#

# In this problem set, you will estimate the CAPM and make predictions for the expected
# return of a stock based on the CAPM. Afterwards, you will estimate the security market
# line. Finally, you will generalize to a multi-factor model and explain portfolio returns
# with the 3 Fama-French factors.
#
# Enjoy!


# Setup
# -----
#
# Import packages for econometric analysis
#
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


# IMPORTANT !!!
#
# For submission, please change the status to 'SOLN', leave it empty while you are working on it.
# Once you set the status to 'SOLN', the code will not work on your machine any more. Don't worry,
# if it worked before setting status = 'SOLN', you're fine.
# If you hand in a file, with status not equal to 'SOLN', it will not count as a regular submission,
# which may lead to an evaluation with 0 points.
#
status = ''
#status = 'SOLN'  # You can also just comment in this line



# Task 1: Read returns of test assets and factors
# -----------------------------------------------

#
# Data description of 16 (4x4) Portfolios formed on market beta & book-to-market
#
# Data from http://www.cfr-cologne.de/english/version06/html/research.php?topic=paper&auswahl=data&v=dl
#
# Beta is estimated relative to DAFOX (CDAX from 2005 onwards) at the end of June of each year T using rolling five year time series regressions with monthly returns.
# To calculate BE/ME in year T we divide book equity for the fiscal year ending in calendar year T-1 by the market value of equity at the end of December in calendar year T-1.
# row (BE/ME), column(beta); 1=low, 4=high
# All of this has already been done and the result can be found in the '16pf_bm_beta.csv' file.
#

# Read-in the test portfolio returns from the provided file '16pf_bm_beta.csv'
# Convert the 'date' column to the datetime data type by using the 'to_datetime' function of the pandas package.
# Note: The date column indicates the date and the month to which the returns of the row correspond.
#
pf_16 =
pf_16['date'] =
pf_16.head()

# Read-in the factors for German market from the provided file 'monthly_factors.csv' (rf is risk free rate):
#   1. rm (Market factor)
#   2. SMB (Fama-French size factor)
#   3. HML (Fama-French value factor)
#   4. WML (Carhart momentum factor)
# Like above, convert the 'date' column to the datetime data type.
#
factors =

factors.head()

# Check of intermediate result (1):
#
# HINT: Check for yourself: pf_16.loc[1, '21'] should be 1.86  (rounded to 2 digits)
#       Check for yourself: factors.loc[4, 'SMB'] should be -0.87  (rounded to 2 digits)
if status == 'SOLN':
    Test.assertEquals(str(np.round(pf_16.loc[5, '32'], 2))[0:4], str(checker_results.loc[1, 'result']), 'Check 1 failed')
    Test.assertEquals(str(np.round(factors.loc[20, 'HML'], 2))[0:4], str(checker_results.loc[2, 'result']), 'Check 2 failed')



# Task 2: Capital Asset Pricing Model
# -----------------------------------

# Estimate the Capital Asset Pricing Model for the '11' and '44' portfolios
#

# First, calculate excess returns and add a constant column (called 'const'), that is fully filled with 1.
# Note: X should be a DataFrame. X.columns should be ['const', 'rm_excess']. Make sure the order is correct!
#       This is very important!
#
X  =

X.columns
X.head()

# Calculate excess returns for the '11' portfolio.
# Note: Y should be a DataFrame. Y.columns should be ['11_excess']
#
Y =
Y.head()



# Set up the OLS model object for the '11' portfolio following the CAPM model:
# r_{11} - r_f = \alpha + \beta * (r_M - r_f) + error_{11}
# Use the OLS class of the statsmodels package, which can be accessed via sm.OLS
#
model_11 =

# Run the OLS parameter estimation by calling the 'fit' function of the OLS object.
#
results_11 =

# Print the details for the model fit by calling the 'summary' function of the OLS
# result object.
#
...


# Redo exercise above for portfolio '44'
#
Y =
Y.head()

model_44   =
results_44 =
...

# Forecasting using the CAPM
#   We assume an expected market risk premium of 6%
#   The CAPM assumes it prices stock returns perfectly
#     => alpha = 0
#     => Exp Ret = beta * Exp MRP
# Note: We use our estimation from above, but only the beta estimate.
# Q: Why did we estimate a non-zero alpha above, but now use alpha = 0?
#
EXP_MARKET_EXCESS_RETURN = 6

# Calculate the expected return for portfolios '11' and '44'.
# Assume a riskfree rate of 2% and an expected market excess return of 6%.
# Hint: You can access your parameter estimates via the 'params' attribute of the OLS result objects.
# Note: exp_ret_11 and exp_ret_44 should be scalars (not pandas or numpy DataFrames/arrays/etc of size 1x1 or similar)
#
RISKFREE = 2
exp_ret_11 =
exp_ret_44 =

# Check of intermediate result (2):
#
# Hint: Check for yourself: results_11.params[0] should be 0.0012 (rounded to 4 digits)
#       Check for yourself: exp_ret_44 should be 7.48 (rounded to 2 digits)
if status == 'SOLN':
    Test.assertEquals(str(np.round(results_11.params[1], 3))[0:5], str(checker_results.loc[3, 'result']), 'Check 3 failed')
    Test.assertEquals(str(np.round(results_44.params[1], 3))[0:5], str(checker_results.loc[4, 'result']), 'Check 4 failed')
    Test.assertEquals(str(np.round(exp_ret_11, 3))[0:5], str(checker_results.loc[5, 'result']), 'Check 5 failed')

plt.scatter(X['rm_excess'], Y['44_excess'], label = ['return observations'])
x = np.linspace(X['rm_excess'].min(), X['rm_excess'].max(), 30)
y = results_44.params[0] + results_44.params[1] * x
plt.plot(x, y, label = 'CAPM-implied expected returns', color = 'black')
plt.legend()
plt.xlabel("Market excess return")
plt.ylabel("Excess return of portfolio 44")
plt.show()



# Task 3: Estimation of the Security Market Line
# ---------------------------------------------------

# Iterate over all columns in the pf_16 DataFrame (except the 'date' column). Each column contains
# the returns of another portfolio. For each portfolio, calculate excess returns and regress them
# on the market's excess returns as you did above. Save the portfolio's beta estimate in the 'betas'
# DataFrame, e.g. like this: betas.loc['33', 'beta'] = ...
#
# Note: We are only interested in the beta estimate. You don't need to save the alpha estimate.
# Hint: Use a for loop to iterate over the portfolio return columns and fill the betas DataFrame
#
portfolios = ['11', '12', '13', '14', '21', '22', '23', '24', '31', '32', '33', '34', '41', '42', '43', '44']
betas = pd.DataFrame(index = portfolios, columns = ['beta'], dtype = np.float64)

for ...

# Now we enter the second stage of the estimation of the securities market line. Recall the
# equation for the securities market line:
#                  r_i - r_f = \gamma * \beta_i + error_i
# You now have beta estimates for each portfolio. You also have monthly (excess) return observations
# for each portfolio. Hence, we can even estimate gamma for each month separately:
#                  r_{i,t} - r_{f,t} = \gamma_t * beta_i + error_{i,t},
# where t indicates the point of time.
# Iterate over all points of time in the pf_16 DataFrame and estimate gamma_t with an OLS
# regression. For each date, store the estimate in the 'gammas' DataFrame, e.g. like this:
# gammas.loc[0, 'gamma'] = ...
# gammas.loc[1, 'gamma'] = ... etc.
#
# Note: Unlike with the CAPM, do NOT add a constant to the regression when you estimate gamma.
# Hint: Again, use a for loop to iterate over the time points of the pf_16 DataFrame.
# Hint: Make sure to regress the excess return of a portfolio on the beta of the SAME portfolio.
#
gammas = pd.DataFrame(columns = ['date', 'gamma'])
gammas['date'] = pf_16['date'].copy()

...

# Calculate the average gamma
# Note: exp_ret_11 and exp_ret_44 should be scalars (not pandas or numpy DataFrames/arrays/etc of size 1x1 or similar)
#
avg_gamma =

# Check of intermediate result (3):
#
# Hint: Check for yourself: betas.loc['12', 'beta'] should be 0.481 (rounded to 3 digits)
#       Check for yourself: gammas.loc[167, 'gamma'] should be -0.67 (rounded to 2 digits)
if status == 'SOLN':
    Test.assertEquals(str(np.round(betas.loc['43', 'beta'], 3))[0:5], str(checker_results.loc[6, 'result']), 'Check 6 failed')
    Test.assertEquals(str(np.round(gammas.loc[53, 'gamma'], 3))[0:6], str(checker_results.loc[7, 'result']), 'Check 7 failed')
    Test.assertEquals(str(np.round(avg_gamma, 3))[0:5], str(checker_results.loc[8, 'result']), 'Check 8 failed')

# Plot the securities market line
plt.scatter(betas['beta'], np.mean(pf_16[portfolios], axis = 0), label = 'Mean portfolio return')
plt.scatter([0], [np.mean(factors['rf'])], label = 'Riskfree investment')
x = np.linspace(0, betas['beta'].max(), 10)
y = np.mean(factors['rf']) + avg_gamma * x
plt.plot(x, y, label = 'Securities Market Line')
plt.legend()
plt.xlabel("Portfolio Beta")
plt.ylabel("Expected Portfolio Return")
plt.show()

# Plot the time series of the market price of risk
plt.plot(gammas['date'], gammas['gamma'])
plt.title("Market Price of Risk")
plt.xlabel("Date")
plt.show()



# Task 4: Fama-French 3 Factor Model
# ----------------------------------

# Estimate the Fama-French 3 Factor Model for portfolios '11' and '44'
#

# First, calculate market excess return (depending on what you did in between, you have this variable
# already from above). To be sure, we recalculate it here. Again, add a column 'const' that is filled
# with 1.
# Note: X should be a DataFrame. X.columns should be ['rm_excess', 'const']
#
X =
X.head()


# Now, add the two other factors to X (columns 'SMB' and 'HML' in X and factors). You do not need to subtract the
# riskfree rate from these factors. Make sure the order of the columns in X is
# X.columns = ['const', 'rm_excess', 'SMB', 'HML']
# Q: Why do we not need to subtract the riskfree rate?
#
...
X.head()


# Calculate excess returns for the '11' portfolio. Again, you already did that above, but depending
# on what happened in between, maybe Y is something different now. Recalculate it here.
# Note: Y should be a DataFrame. Y.columns should be ['11_excess']
#
Y =
Y.head()


# Now, estimate the FF3 model. Again, use OLS with the additional 'SMB' and 'HML' columns now.
#
model_11   =
results_11 =

# Inspect the OLS estimates by calling the 'summary' function of the OLS result object.
#
...

# Repeat the exercise above for the '44' portfolio and inspect the Fama-French OLS estimates
# of the '44' portfolio.
#
Y =
model_44   =
results_44 =
...

# Check of intermediate result (4):
#
# Hint: Check for yourself: results_11.params[2] should be 0.461 (rounded to 3 digits)
#       Check for yourself: results_44.params[2] should be 0.793 (rounded to 3 digits)
if status == 'SOLN':
    Test.assertEquals(str(np.round(results_11.params[3], 3))[0:6], str(checker_results.loc[9, 'result']), 'Check 9 failed')
    Test.assertEquals(str(np.round(results_44.params[3], 3))[0:5], str(checker_results.loc[10, 'result']), 'Check 10 failed')
