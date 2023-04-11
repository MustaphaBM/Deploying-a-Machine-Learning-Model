# Model Card

## Model Details
Random forest model that uses the hyperparameters below :

    - criterion : gini
    - n_estimators : 30
    - max_depth : 10
## Intended Use
To predict whether an employee's salary is greater than $50k or not  

## Training Data
Census Income dataset  : 
80% of the dataset above is used to train this model 
## Evaluation Data
Census Income dataset : https://archive.ics.uci.edu/ml/datasets/census+income
20% of the dataset above used for evaluation
## Metrics
- Precision : 0.808
- Recall : 0.536
- FBeta : 0.644
## Ethical Considerations
The dataset is from UCI machine learning repository. it's public but it should be referenced 
https://archive.ics.uci.edu/ml/datasets/census+income
## Caveats and Recommendations
The names of the columns contain white spaces, we should consider remove the extra spaces