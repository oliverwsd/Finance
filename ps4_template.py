#
# ================================================
# Empirical Finance WS 19/20
# Problem Set 4, Week 7
# Modeling of Factor Returns
# ================================================
#
# Prepared by Elmar Jakobs & Simon Walther
#

# In this problem set, you will estimate an ARMA(1, 1) model to predict
# factor returns. You will then create return forecasts and simulate
# the model to obtain the model-implied distribution of factor returns.
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


# Read-in the factors for German market from the provided file 'monthly_factors.csv' (rf is risk free rate):
#   1. rm (Market factor)
#   2. SMB (Fama-French size factor)
#   3. HML (Fama-French value factor)
#   4. WML (Carhart momentum factor)
# Like above, convert the 'date' column to the datetime data type.
#
factors         = pd.read_csv('monthly_factors.csv')
factors['date'] = pd.to_datetime(factors['date'], format ='%Y%m')
factors['rm_excess'] = factors['rm'] - factors['rf']
factors = factors[['date', 'rm_excess', 'rf']]  # We'll only use market excess returns and riskfree rate here
factors[['rm_excess', 'rf']] /= 100  # Bring numbers to actual values
factors.head()


# We will estimate an ARMA(1, 1) model on market excess returns and the riskfree rate:
#
#    r_t = \alpha + \beta * r_{t-1} + \epsilon_t + \theta * \epsilon_{t-1}   (1)
#
# This means: We need to come up with estimates for the parameters \alpha, \beta and \theta
#
# The problem is: \epsilon_{t-1} is not observed, we only have observations for r_t, r_{t-1}.
# We will follow a two-pass estimation to circumvent this problem. We first estimate the
# AR part of that model via OLS:
#
#    r_t = \alpha + \beta * r_{t-1} + \tilde{\epsilon}_t                      (2)
#
# Given this model, we can get a time series of \tilde{\epsilon}_t. With that, we can now
# estimate (1) via OLS, by just setting \epsilon_{t-1} = \tilde{\epsilon}_{t-1}.


# Task 1: Estimate AR model part (Step 1)
# ---------------------------------------

# Estimate the model (2) for market excess returns via OLS using the statsmodels package.
#
# Hint: If you are insecure what to do, write down the regression formulas for the factor
#       models of the last problem set and compare.
# Note: 'ar_model_rm_excess' should contain the OLS regression result object (after fitting)
#
# IMPORTANT: Use the constant as first variable in the regression. This also applied to the
#            next regression!

ar_model_rm_excess =
print(ar_model_rm_excess.summary())


# Repeat the analysis above for the riskfree rate.
#
# Note: 'ar_model_rf' should contain the OLS regression result object

ar_model_rf =
print(ar_model_rf.summary())


# Check of intermediate result (1):
#
# HINT: Check for yourself: ar_model_rm_excess.params[0] should be 0.0038  (rounded to 4 digits)
#       Check for yourself: ar_model_rf.params[0] should be 0.00007  (rounded to 5 digits)
if status == 'SOLN':
    Test.assertEquals(str(np.round(ar_model_rm_excess.params[1], 3))[0:5], str(checker_results.loc[1, 'result']), 'Check 1 failed')
    Test.assertEquals(str(np.round(ar_model_rf.params[1], 3))[0:5], str(checker_results.loc[2, 'result']), 'Check 2 failed')


# Task 2: Estimate full ARMA model (Step 2)
# -----------------------------------------

# Compute the time series of error "observations" \tilde{\epsilon}_t for the market excess return.
# Afterwards, estimate the parameters of model (1) via OLS using the statsmodels package for market
# excess returns.
#
# Note: 'arma_model_rm_excess' should contain the OLS regression result object (after fitting)
# IMPORTANT: The order of the variables in the model should be (constant, r_{t-1}, \epsilon_{t-1}).
#            We expect the parameter estimates to follow this order. This also applies to the next
#            regression!

arma_model_rm_excess =
print(arma_model_rm_excess.summary())

# Is the parameter for the AR part, beta, significant to the 5% confidence level in the ARMA model for
# market excess returns?
#
# If so, set ar_significant_rm_excess = 'yes'. Otherwise, set ar_significant_rm_excess = 'no'.
ar_significant_rm_excess =

# Repeat the ARMA model estimation for the riskfree rate.
#
# Note: 'arma_model_rf' should contain the OLS regression result object (after fitting)

arma_model_rf =
print(arma_model_rf.summary())

# Is the parameter for the AR part, beta, significant to the 5% confidence level in the ARMA model for
# the riskfree rate?
#
# If so, set ar_significant_rf = 'yes'. Otherwise, set ar_significant_rf = 'no'.
ar_significant_rf =


# Check of intermediate result (2):
#
# HINT: Check for yourself: arma_model_rm_excess.params[1] should be -0.072  (rounded to 3 digits)
#       Check for yourself: arma_model_rf.params[1] should be 0.988  (rounded to 3 digits)
if status == 'SOLN':
    Test.assertEquals(str(np.round(arma_model_rm_excess.params[2], 3))[0:5], str(checker_results.loc[3, 'result']), 'Check 3 failed')
    Test.assertEquals(str(np.round(arma_model_rf.params[2], 3))[0:6], str(checker_results.loc[4, 'result']), 'Check 4 failed')
    Test.assertEquals(ar_significant_rm_excess, str(checker_results.loc[5, 'result']), 'Check 5 failed')
    Test.assertEquals(ar_significant_rf, str(checker_results.loc[6, 'result']), 'Check 6 failed')


# Task 3: Predict returns
# -----------------------

# Predict market excess returns based on the ARMA model. That means, given information
# in t, predict the value of the market excess return in t+1. Use the epsilon estimates
# that you also used in the parameter estimation above. Do only predict those returns
# for which we have observations (i.e. do not make the prediction at the last point of
# the time series, as we do not have an observation after that).

E_rm_excess =
# Hint: E_rm_excess.shape should be (616,)


# Repeat the same for the riskfree rate and predict it using its ARMA model parameters.
# Again, do not make a prediction for the last point of the time series and use the
# epsilon estimates that you used in the estimation above.

E_rf =
# Hint: E_rf.shape should be (616,)


# Plot expectations and realization
fig, ax = plt.subplots(2, 1)
ax[0].plot(factors['date'].iloc[2:].values, factors['rm_excess'].iloc[2:].values, label = 'Realizations')
ax[0].plot(factors['date'].iloc[2:].values, E_rm_excess, label = ['Expectations'])
ax[0].set_title("Market excess return")
ax[0].legend()
ax[1].plot(factors['date'].iloc[2:].values, factors['rf'].iloc[2:].values, label = 'Realizations')
ax[1].plot(factors['date'].iloc[2:].values, E_rf, label = ['Expectations'])
ax[1].set_title("Riskfree rate")
ax[1].legend()
plt.show()


# What is the unconditional expectation of the riskfree rate, that is implied
# by the ARMA model?
#
# Hint: Derive the formula for the unconditional expectation at first.
unconditional_E_rf =


# Check of intermediate result (3):
#
# HINT: Check for yourself: E_rm_excess[23] should be 0.0093  (rounded to 4 digits)
#       Check for yourself: E_rf[23] should be 0.0028  (rounded to 4 digits)
if status == 'SOLN':
    Test.assertEquals(str(np.round(E_rm_excess[43], 4))[0:7], str(checker_results.loc[7, 'result']), 'Check 7 failed')
    Test.assertEquals(str(np.round(E_rf[43], 4))[0:6], str(checker_results.loc[8, 'result']), 'Check 8 failed')
    Test.assertEquals(str(np.round(unconditional_E_rf, 4))[0:6], str(checker_results.loc[9, 'result']), 'Check 9 failed')


# Task 4: Simulate an ARMA(1, 1) model
# ------------------------------------

sigma2_epsilon = 0.00257654   # Do not change this line

# Simulate the ARMA(1, 1) model for market excess returns. Draw \epsilon_t from a
# normal distribution with mean 0 and variance 'sigma2_epsilon', using numpy's
# np.random.normal(...) function. Then, plug these draws into the model equation (1).
np.random.seed(100)  # Do not change this line. Makes the generated random number repeatable. Every time you call
                     # this line, the same sequence of random number will be drawn afterwards.
rm_excess_sim = np.zeros((factors.shape[0],))
rm_excess_sim[0] = 0   # Do not change this line. Consider this as r_0 = 0.

...

# Plot histograms of actual and simulated market excess returns
fig, ax = plt.subplots(1, 2)
ax[0].hist(factors['rm_excess'].values, bins = 50, density = True)
ax[0].set_title("Observed market returns")
ax[1].hist(rm_excess_sim, bins = 50, density = True)
ax[1].set_title("Simulated market returns")
plt.show()

# Check of intermediate result (4):
#
# HINT: Check for yourself: rm_excess_sim[23] should be -0.0406  (rounded to 4 digits)
if status == 'SOLN':
    Test.assertEquals(str(np.round(rm_excess_sim[43], 3))[0:6], str(checker_results.loc[10, 'result']), 'Check 10 failed')
    Test.assertEquals(str(np.round(rm_excess_sim[44], 3))[0:5], str(checker_results.loc[11, 'result']), 'Check 11 failed')
