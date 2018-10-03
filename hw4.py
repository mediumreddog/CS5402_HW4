from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# Load IRIS dataset
iris = load_iris()

# Decision Tree Classifier
decTree = DecisionTreeClassifier(random_state=0)
print "Decision"
decTreeScores = cross_val_score(decTree, iris.data, iris.target, cv=5)
decTreeF1Scores = cross_val_score(decTree, iris.data, iris.target, cv=5, scoring='f1_macro')
print("Decision Tree Accuracy: %0.4f (+/- %0.4f)" % (decTreeScores.mean(), decTreeScores.std() * 2))
print("Decision Tree F1: %0.4f (+/- %0.4f)" % (decTreeF1Scores.mean(), decTreeF1Scores.std() * 2))

print

# kNN Classifier
for i in range(1, 6):
  knn = KNeighborsClassifier(n_neighbors=i)
  print "kNN: n=" + str(i)
  knnScores = cross_val_score(knn, iris.data, iris.target, cv=5)
  knnF1Scores = cross_val_score(knn, iris.data, iris.target, cv=5, scoring='f1_macro')
  print("kNN Accuracy: %0.4f (+/- %0.4f)" % (knnScores.mean(), knnScores.std() * 2))
  print("kNN F1: %0.4f (+/- %0.4f)" % (knnF1Scores.mean(), knnF1Scores.std() * 2))
