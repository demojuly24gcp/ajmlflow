# classification model --> create a classification model 

# which dataset are we going to train . 

# simplicity --> create a synthetic data on which I will train a ML model 

# it is a Supervised Machine Learning model in which the target feature are categorical varables. 
# 
# classification data
# 
#  

from sklearn.datasets import make_classification

# Step 1: Create an imbalanced binary classification dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=2, n_redundant=8, 
                           weights=[0.9, 0.1], flip_y=0, random_state=42)



