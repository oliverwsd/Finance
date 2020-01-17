#
# ================================================
# Empirical Finance WS 19/20
# Problem Set 5, Week 10
# A GARCH model for equity returns
# ================================================
#
# Prepared by Simon Walther
#

# In this problem set, you will estimate an AR(1)-GARCH(1,1) model to predict
# factor returns and their volatility. You will compare the volatility forecast
# of the GARCH model with the vol forecast of a simpler ARCH(1) model.
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
import math
import scipy.optimize


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



# Task 1: Estimate AR(1)-ARCH(1) via 2-pass estimation
# ----------------------------------------------------

# We start by estimating the AR(1)-ARCH(1) via a two-pass estimation approach, that is,
# we first estimate the AR model via OLS, then calculate the residuals of that AR(1)
# model and use them to estimate the ARCH(1) part via OLS.
#
# Estimate the AR(1) model for market excess returns via OLS.
#
# Note: 'ar_model' should contain the OLS regression result object (after fitting)
#
# IMPORTANT: Use the constant as first variable in the regression. This also applies to the
#            next regression!

ar_model =
print(ar_model.summary())


# Now, we attack the ARCH(1) part. First, compute the residuals of the AR(1) model that
# you just fitted.
#
ar_model_resid =

# Recall, an ARCH(1) model reads
#     \sigma_t^2 = a + b * \epsilon_{t-1}^2
# with \sigma_t^2 being the variance of \epsilon_t and \epsilon_t is the residual of
# the AR(1) model. We now use \epsilon_t^2 as a very rough measurement for
# \sigma_t^2 (i.e. just replace \sigma_t^2 in the equation above with \epsilon_t^2)
# and estimate the model via OLS.
#
# Please do so next.
#
# Note: 'arch_model' should contain the OLS regression result object (after fitting)
#

arch_model =
print(arch_model.summary())


# Check of intermediate result (1):
#
# HINT: Check for yourself: ar_model.params[0] should be 0.0038 (rounded to 4 digits)
#       Check for yourself: arch_model.params[0] should be 0.0023  (rounded to 4 digits)
if status == 'SOLN':
    Test.assertEquals(str(np.round(ar_model.params[1], 3))[0:5], str(checker_results.loc[1, 'result']), 'Check 1 failed')
    Test.assertEquals(str(np.round(ar_model_resid[3], 3))[0:6], str(checker_results.loc[2, 'result']), 'Check 2 failed')
    Test.assertEquals(str(np.round(arch_model.params[1], 3))[0:5], str(checker_results.loc[3, 'result']), 'Check 3 failed')


# Task 2: MLE estimation of AR(1)-ARCH(1) model
# ---------------------------------------------

# The 2-pass OLS estimation does not generally lead to parameter estimates that maximize
# the (log-)likelihood of the model. Such parameters must be identified via numerical
# optimization. For that, we assume that excess returns r_t are distributed
# normally, with mean \mu_t and volatility \sigma_t. Recall that the likelihood equation
# for the normal distribution reads:
#
#   L(r_t) = (1 / (2 * \pi * \sigma_t^2)^0.5) * e^(-(r_t - \mu_t)^2 / (2 * \sigma_t^2))
#
# This is the likelihood for a single r_t. To get the likelihood of the full time
# series of r_t, we have to multiply the probabilities:
#
#   L(r_{1:T}) = \prod_{t=1}^T L(r_t)
#
# Since a product is hard to compute precisely (all probabilities are < 0, so the product
# will become very small), we just take the log of the likelihood:
#
#   log L(r_{1:T}) = \sum_{t=1}^T log L(r_t)
#
# Before you continue coding, take a pen and paper and derive the log-likelihood formula
# for the AR(1)-ARCH(1) model.
#
# Hint: Recall, in an AR(1) model, the mean prediction is \mu_t = E_{t-1}(r_t) =
#       E(\alpha + \beta * r_{t-1} + \epsilon_t). The variance prediction of the ARCH(1)
#       model is \sigma_t^2 = E_{t-1}(\epsilon_t^2) = E(a + b * \epsilon_{t-1}^2).
#       Plug these formulas into the equations above. Since you make each forecast for time t
#       in t-1, you can use everything that is known in t-1 as an observed variable (i.e. not
#       a random number any more.


# Once you derived the log-likelihood formula, implement it in the function below. The function
# should calculate the log-likelihood of the observed excess returns, given a set of parameters.
#
# Note: Since you need the first AR(1)-residual \epsilon_1 for the variance forecast, skip
#       the very first return in the actual log-likelihood calculation.
# Hint: You can get \pi as math.pi
#
def loglikelihood_ar_arch(parameters):   # Parameters is a list of model parameters, here: [\alpha, \beta, a, b]
    ar_alpha = parameters[0]
    ar_beta = parameters[1]
    arch_a = parameters[2]
    arch_b = parameters[3]

    loglikeli =

    return -loglikeli  # We return the negative log-likelihood. This is intentional, you'll see in a moment, why.

# Now you are ready to optimize the loglikelihood. Python provides a number of out-of-the-box
# optimization algorithms in the scipy package. We will use the function scipy.optimize.minimize(...)
# to perform the optimization. As its name suggests, the function minimizes some value. Since
# we want to maximize the loglikelihood, we'll just minimize the negative loglikelihood.
# That why we let the function above return the negative log-likelihood.
#
# Take a look at the function's documentation, especially at the examples on the bottom, to
# see how to use the function:
#   https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
#
# Besides our objective function (which is the log-likelihood calculation function above), we
# only need to define a starting value for the parameters to get the optimization started.
# We just use the parameters estimates from the 2-pass OLS estimation above as starting
# values.
#
# Hint: Use method = 'Nelder-Mead' in the scipy.optimize.minimize(...) function.
# Hint: ar_arch_params_start should be a list with 4 elements.
#

ar_arch_params_start =
ar_arch_params = scipy.optimize.minimize(

# Given the optimal model parameters, compute the volatility (!) forecast of the model.
#

arch_vol_forecasts =           # Hint: type(arch_vol_forecasts) should be np.ndarray and arch_vol_forecasts.shape should be (616,)


# Now, let's plot the vol forecasts
fig, ax = plt.subplots(2, 1)
ax[0].plot(factors['date'].iloc[2:], factors['rm_excess'].iloc[2:])
ax[0].set_title("Market excess returns")
ax[1].plot(factors['date'].iloc[2:], arch_vol_forecasts)
ax[1].set_title("One-day ahead daily vol forecast (ARCH model)")
plt.show()

# Check of intermediate result (2):
#
# HINT: Check for yourself: loglikelihood_ar_arch([0.01, 0.2, 0.01, 0.1]) should be -766.19 (rounded to 2 digits)
#       Check for yourself: ar_arch_params.x[0] should be 0.0045  (rounded to 4 digits)
#       Check for yourself: arch_vol_forecasts[0] should be 0.0514 (rounded to 4 digits)
if status == 'SOLN':
    Test.assertEquals(str(np.round(loglikelihood_ar_arch(ar_arch_params_start), 3))[0:8], str(checker_results.loc[4, 'result']), 'Check 4 failed')
    Test.assertEquals(str(np.round(ar_arch_params.x[3], 3))[0:5], str(checker_results.loc[5, 'result']), 'Check 5 failed')
    Test.assertEquals(str(np.round(arch_vol_forecasts[100], 3))[0:5], str(checker_results.loc[6, 'result']), 'Check 6 failed')


# Task 3: MLE estimation of AR(1)-GARCH(1, 1) model
# -------------------------------------------------

# The GARCH model is similar to the ARCH model, but also makes the variance forecast for
# tomorrow depend on the variance forecast of today:
#    \sigma_t^2 = a + b * \epsilon_{t-1}^2 + c * \sigma_{t-1}^2
#
# Write a function that calculates the time series for \sigma_t based on given parameters
# a, b, c, an initial value for \sigma^2 and a time series of epsilon_t.
#
# Hint: Use a for-loop.
# Note: Do not create a \sigma^2 estimate based on the last \epsilon observation. This would
#       create a forecast for the first return after our observed data sample. We don't want that.
#

def garch_variance(a, b, c, sigma_initial, epsilon):
    sigma2 =
    return sigma2

test_garch_variance =  garch_variance(0.01, 0.15, 0.8, 0.2, ar_model_resid)    # Hint: type(test_garch_variance) should be np.ndarray
                                                                               # Hint: test_garch_variance.shape should be (616,)

# How write a function that calculates the log-likelihood of excess returns in an AR(1)-GARCH(1,1) model
# based on the variance forecast of the GARCH model. Follow the same reasoning
# and procedure as for the ARCH model above.
#
def loglikelihood_ar_garch(parameters):
    ar_alpha = parameters[0]
    ar_beta = parameters[1]
    garch_a = parameters[2]
    garch_b = parameters[3]
    garch_c = parameters[4]
    garch_initial_sigma = parameters[5]

    loglikeli =

    return -loglikeli  # We return the negative log-likelihood. This is intentional, you'll see in a moment, why.

# Now, again, use the scipy.optimize.minimize function to find the optimal
# parameters of the AR(1)-GARCH(1,1) model.
#
# Again, use method = 'Nelder-Mead' in the scipy.optimize.minimize(...) function.
#
ar_garch_params_start = [ar_model.params[0], ar_model.params[1], 0.0001, 0.15, 0.8, np.var(factors['rm_excess'])]
ar_garch_params =

# Given the optimal model parameters, compute the volatility (!) forecast of the GARCH(1,1) model.
#

garch_vol_forecasts =       # Hint: type(garch_vol_forecasts) should be np.ndarray and garch_vol_forecasts.shape should be (616,)

# Now, let's plot the vol forecasts
fig, ax = plt.subplots(2, 1)
ax[0].plot(factors['date'].iloc[2:], factors['rm_excess'].iloc[2:])
ax[0].set_title("Market excess returns")
ax[1].plot(factors['date'].iloc[2:], arch_vol_forecasts, label = 'ARCH model vol forecast')
ax[1].plot(factors['date'].iloc[2:], garch_vol_forecasts, label = 'GARCH model vol forecast')
ax[1].set_title("One-day ahead daily vol forecast")
ax[1].legend()
plt.show()


# Check of intermediate result (3):
#
# HINT: Check for yourself: test_garch_variance[0] should be 0.1706 (rounded to 4 digits)
#       Check for yourself: loglikelihood_ar_garch([0.001, 0.15, 0.0002, 0.3, 0.65, 0.001]) should be -974.81 (rounded to 2 digits)
#       Check for yourself: ar_garch_params.x[0] should be 0.0038  (rounded to 4 digits)
#       Check for yourself: garch_vol_forecasts[0] should be 0.1086 (rounded to 4 digits)
if status == 'SOLN':
    Test.assertEquals(str(np.round(test_garch_variance[100], 3))[0:5], str(checker_results.loc[7, 'result']), 'Check 7 failed')
    Test.assertEquals(str(np.round(loglikelihood_ar_garch(ar_garch_params_start), 3))[0:8], str(checker_results.loc[8, 'result']), 'Check 8 failed')
    Test.assertEquals(str(np.round(ar_garch_params.x[3], 3))[0:5], str(checker_results.loc[9, 'result']), 'Check 9 failed')
    Test.assertEquals(str(np.round(garch_vol_forecasts[100], 3))[0:5], str(checker_results.loc[10, 'result']), 'Check 10 failed')

