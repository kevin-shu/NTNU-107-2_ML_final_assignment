
# How to run the program
In the terminal, switch to the folder of the project, and then:
``` 
# Install required modules: 
pip install -r requirements.txt

# Run the SVM training and 10-fold validation:
python source/svm.py

# Run the K-Means clustering and 10-fold validation:
python source/kmeans.py
```

# Requirements:

## Check if the accuracy of the learned classifier remains unchanged when removing a feature.
After observing the correlation between each features and target using `Pandas.DataFrame.corr()`, I found out that most of the continuous data have very poor correlation with the target. So first I drop all the continuous features like `BILL_AMT` and `PAY_AMT`, and the result almost remain the same as before.  
Then I pick `PAY_0`, which has highest correlation with target among categorical features, to train SVM. And the result is almost the same. So I suggest that we could use only `PAY_0` to train SVM in this case.  
Here are the result:
* Linear SVM's accuracy using **all features**: 0.821103
* Linear SVM's accuracy using only `PAY_0`: 0.819336

## Compare the performance of linear and polynomial basis functions of your SVMs (using a 10-fold cross-validation)
* **Linear** SVM's accuracy using only `PAY_0`: 0.819336
* **Polynomial** SVM's accuracy using only `PAY_0`: 0.819403


## K-Means clustering accuracy:
After observing the correlation between each features and target, I chose "LIMIT_BAL", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6" to do the clustering, here are the result:
* Kmeans' with 2 group accuracy: 0.778800  
* Kmeans' with 4 group accuracy: 0.793767  
* Kmeans' with 8 group accuracy: 0.798433  
* Kmeans' with 16 group accuracy: 0.802933
* Kmeans' with 32 group accuracy: 0.811500  


# References:
* https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
* https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
* https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
* http://stamfordresearch.com/k-means-clustering-in-python/