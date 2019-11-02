from sklearn.datasets import load_digits 
from sklearn.ensemble import RandomForestClassifier
digits, labels = load_digits(return_X_y=True)
test_digits=digits[:200]
test_labels=labels[:200]
validation_digits=digits[200:400]
validation_labels=labels[200:400]
training_digits=digits[400:]
training_labels=labels[400:]
for split in ["gini", "entropy"]:
    for n_trees in [5,10,20,100]:
        for tree_depth in [2,5,10,None]:
            clf=RandomForestClassifier(n_estimators=n_trees, criterion=split, max_depth=tree_depth) 
            clf.fit(training_digits, training_labels)
            print("Split criterion", split, ", Nr trees:", n_trees, ", tree depth", tree_depth)
            print("Acc:",clf.score(validation_digits,validation_labels))

