import pandas
import dataset
import statsmodels.formula.api as smfapi
import joblib

# load data in and use pandas to turn it into a DataFrame
dataset1 = pandas.DataFrame(dataset.data)

# Fit (train) a simple model that finds a linear relationship between boot size and harness size. 
# We can use this model later to predict a dog's boot size, given their harness size.
model = smfapi.ols(formula = "boot_size ~ harness_size", data = dataset1).fit()
print("Model trained!")

# Our model is ready to use, but we don't need it yet. Let's save it to disk.
model_filename = './dog_boot_model.pkl'
joblib.dump(model, model_filename)
print("Model saved!")



# Let's write a function that loads and uses our model
def load_model_and_predict(harness_size):
    '''
    This function loads a pretrained model. It uses the model
    with the customer's dog's harness size to predict the size of
    boots that will fit that dog.

    harness_size: The dog harness size, in cm 
    '''
    # Load the model from file and print basic information about it
    loaded_model = joblib.load(model_filename)
    print("We've loaded a model with the following parameters:")
    print(loaded_model.params)

    # Prepare data for the model
    inputs = {"harness_size":[harness_size]} 

    # Use the model to make a prediction
    predicted_boot_size = loaded_model.predict(inputs)[0]
    return predicted_boot_size


# Practice using our model
predicted_boot_size = load_model_and_predict(45)
print("Predicted dog boot size:", predicted_boot_size)