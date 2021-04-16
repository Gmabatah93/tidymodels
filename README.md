# Tidymodels
<img src="Images/tidymodels.PNG" width="700">

# Data Resampling:
Simple Random Resampling: is appropriate in many cases.
- when there is severe class imbalance

## Resampling
<img src="Images/resampling.PNG" width="500">

### Cross-Validation
<img src="Images/cross.PNG" width="300">
<img src="Images/3cross.PNG" width="300">

 Data is randomly partitioned into V sets of roughly equal size (called the “folds”).

#### Repeated
_One way to reduce noise is to gather more data. For cross-validation, this means averaging more than V statistics._

#### Leave-One-Out

#### Monte Carlo
_Allocates a fixed proportion of data to the assessment set. The difference is that, for MCCV, this proportion of the data is randomly selected each time. This results in assessment sets that are not mutually exclusive. To create these resampling objects:_

### Validation Set
<img src="Images/validation.PNG" width="300">

_single partition that is set aside to estimate performance, before using the test set_

### Bootstraping
<img src="Images/bootstrap.PNG" width="300">

_Was originally invented as a method for approximating the sampling distribution of statistics whose theoretical properties are intractable. A bootstrap sample of the training set is a sample that is the same size as the training set but is drawn with replacement. This means that some training set data points are selected multiple times for the analysis set. Each data point has a 63.2% chance of inclusion in the training set at least once. The assessment set contains all of the training set samples that were not selected for the analysis set (on average, with 36.8% of the training set). When bootstrapping, the assessment set is often called the “out-of-bag” sample._

#### Functions

Function | Description | Parameters
--- | --- | ---
initial_split() | creates a single binary split of the data into a training set and testing set | data, prop, strata, breaks, lag
training() | takes the 1st prop of samples from _initial_split()_ |
testing() | takes the 2st prop of samples from _initial_split()_ |
vfold_cv() | randomly splits the data into V groups of roughly equal size (called "folds") | data, v, repeats, strata, breaks
mc_cv() | One resample of Monte Carlo cross-validation takes a random sample (without replacement) of the original data set to be used for analysis. All other data points are added to the assessment set. | data, prop, times, strata, breaks
validation_split() | takes a single random sample (without replacement) of the original data set to be used for analysis. All other data points are added to the assessment set (to be used as the validation set) | data, prop, strata, breaks
bootstraps() | A bootstrap sample is a sample that is the same size as the original data set that is made using replacement | data, times, strata, breaks

# Feature Engineering
- Correlation between predictors
- missing values
- Distribution
- Center and Scale
- Feature Extraction

#### Functions

Function | Description | Parameters
--- | --- | ---
recipe() | A recipe is a description of what steps should be applied to a data set in order to get it ready for data analysis | formula, data
prep() | Train a Data Recipe | x, training, verbose, retain
bake() | Apply a Trained Data Recipe | object, new_data
step_log() | creates a specification of a recipe step that will log transform data | recipe, base, columns
step_dummy() | dummy variable creation | recipe, one_hot, naming
step_normalize | Center and Scale numeric data | recipe
step_unknown() | Assign missing categories to "unknown" | recipe, new_level
step_other() | creates a specification of a recipe step that will potentially pool infrequently occurring values into an "other" category | recipe, threshold
step_interact() | creates interaction variables | recipe, terms
step_filter() | creates a specification of a recipe step that will remove rows using dplyr::filter(). | recipe, inputs
step_downsample() | Down-Sample a Data Set Based on a Factor Variable | recipe, under_ratio, target, skip
step_upsample() | Up-Sample a Data Set Based on a Factor Variable | recipe, over_ratio, target, skip

# Model Fitting


#### Functions

Function | Description | Parameters
--- | --- | ---
linear_reg() | General Interface for Linear Regression Models | penalty, mixture
logistic_reg() | General Interface for Logistic Regression Models | penalty, mixture
nearest_neighbor() | General Interface for K-Nearest Neighbor Models | neighbors, weight_func, dist_power
mars() | General Interface for MARS | num_terms, prod_degree, prune_method
svm_rbf() | General interface for radial basis function support vector machines | cost, rbf_sigma, margin
decision_tree() | General Interface for Decision Tree Models | cost_complexity, tree_depth, min_n
rand_forest() | General Interface for Random Forest Models | mtry, trees, min_n
boost_tree() | General Interface for Boosted Trees | mtry, trees, min_n, tree_depth, learn_rate, loss_reduction, sample_size, stor_iter
surv_reg() | General Interface for Parametric Survival Models | dist

# Model Evaluation
> Bias is the difference between the true data pattern and the types of patterns that the model can emulate. Many black-box machine learning models have low bias. Other models (such as linear/logistic regression, discriminant analysis, and others) are not as adaptable and are considered high-bias models.

#### Functions

Function | Description | Parameters
--- | --- | ---
metric_set() | Combine metric functions | names of the functions to be included in the metric set
rmse() | Root mean squared error | truth, estimate
rsq() | R squared | truth, estimate
mae() | Mean absolute error | truth, estimate
conf_mat() | Confusion Matrix for Categorical Data | truth, estimate, dnn
accuracy() | Accuracy | truth, estimate
