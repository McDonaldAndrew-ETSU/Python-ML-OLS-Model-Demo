import pandas
import dataset

# load data in and use pandas to turn it into a DataFrame
dataset1 = pandas.DataFrame(dataset.data)

# Print the data
# In normal python we would write
print(dataset1)

# Load a library to do the hard work for us
import statsmodels.formula.api as smfapi

# First, we define our formula using a special syntax
# This says that boot_size is explained by harness_size
formula = "boot_size ~ harness_size"

# Create the model, but don't train it yet 
model = smfapi.ols(formula = formula, data = dataset1)

# Note that we have created our model but it does not 
# have internal parameters set yet
if not hasattr(model, 'params'):
    print("Model selected but it does not have parameters set. We need to train it!")


# OLS (Ordinary Least Squares) models have two parameters (a slope and an offset), but these haven't been set in our model yet. 
# We need to train (fit) our model to find these values so that the model can reliably estimate dogs' boot size based on their harness size.

# Train (fit) the model so that it creates a line that fits our data. 
# We will look at how this method works in a later unit.
fitted_model = model.fit()


# Print information about our model now it has been fit
print("The following model parameters have been found:\n" +
        f"Line Intercept: {fitted_model.params[0]}\n" +
        f"Line slope: {fitted_model.params[1]}\n")

# harness_size states the size of the harness we are interested in
# 52.5 can be changed 
harness_size = { 'harness_size' : [52.5] }

# Use the model to predict what size of boots the dog will fit
approximate_boot_size = fitted_model.predict(harness_size)

# Print the result
print(f"Estimated approximate_boot_size: {approximate_boot_size[0]}")


# So that a computer can understand our objective, we need to provide our goal as code snippet called an objective function (cost function). 
# Objective functions judge whether the model is doing a good job (estimating boot size well) or bad job (estimating boot size badly).

# Data refers to the information that we provide to the model (also known as inputs). 
# In our scenario, this is harness size.
# Data also refers to information that the objective function might need. 
# For example, if our objective function reports whether the model guessed the boot size correctly, it will need to know the correct boot size! 
# This is why in our previous exercise, we provided both harness sizes and the correct answers to the training code.

# During training, the model makes a prediction, and the objective function calculates how well it performed. 
# The optimizer is code that then changes the model’s parameters so the model will do a better job next time.


# It's common to build, train, then use a model while we are just learning about machine learning; 
# but in the real world, we don't want to train the model every time we want to make a prediction.