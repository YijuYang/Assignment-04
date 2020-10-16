import random
import numpy as np
from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig

from pandas import read_csv
from matplotlib import pyplot

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from pandas.plotting import scatter_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from math import exp
from math import floor

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)
print("\n\n-------------------------------------------------")
print ('Part 01: Using original features')
# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.50, random_state=1, shuffle=True)

dtree = DecisionTreeClassifier()
dtree.fit(X_train, Y_train)
prediction1 = dtree.predict(X_validation)
dtree.fit(X_validation, Y_validation)
prediction2 = dtree.predict(X_train)
predictions = np.concatenate([prediction1, prediction2])
Y_test = np.concatenate((Y_validation, Y_train))

print("features:" + ' sepal-length'+' sepal-width'+' petal-length'+' petal-width')
print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, predictions))
print("\nAccuracy Score:")
print(accuracy_score(Y_test, predictions))
print("\n\n-------------------------------------------------")

print("Part 2: Using PCA to transform the original 4 features ")
X = array[:, 0:4]

pca = PCA(4)
pca.fit(X)

X = pca.transform(X)
for row in X:
    row = row[0]

y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.50, random_state=1, shuffle=True)

dtree = DecisionTreeClassifier()
dtree.fit(X_train, Y_train)
prediction1 = dtree.predict(X_validation)
dtree.fit(X_validation, Y_validation)
prediction2 = dtree.predict(X_train)
predictions = np.concatenate([prediction1, prediction2])
Y_test = np.concatenate((Y_validation, Y_train))

print("features:"+' sepal-length-prime'+' sepal-width-prime'+' petal-length-prime'+' petal-width-prime')
print("\nEigenvectors:")
print(pca.components_)
print("\nEigenvalues:")
print(pca.explained_variance_)
pov = pca.explained_variance_[0] / sum(pca.explained_variance_)
print("\nPoV:")
print(pov)
print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, predictions))
print("\nAccuracy Score:")
print(accuracy_score(Y_test, predictions))
print("\n\n-------------------------------------------------")

print("Part 3: Using Simulated Annealing")
X_transformed = X
X = array[:, 0:4]
y = array[:,4]
feature_names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'sepal-length-prime', 'sepal-width-prime', 'petal-length-prime', 'petal-width-prime']
combined_features_orig = np.column_stack((X, X_transformed))

end = True
while(end):
    current_feature_set = combined_features_orig
    for i in range(100):
        temp_arr = current_feature_set
        perturb_num =  floor(random.uniform(0.01, 0.05) * 100)
        if random.randint(0, 100) % 2 == 0:
            for j in range(int(perturb_num)):
                concat_array = []
                concat_array.append(combined_features_orig[:, random.randint(0, 7)])
                concat_array = np.array(concat_array)
                concat_array = concat_array.reshape((150, 1))
                temp_arr = np.append(temp_arr, concat_array, 1)
        else:
            for j in range(int(perturb_num)):
                rand_num = 0
                if np.size(temp_arr, 1) <= 1:
                    rand_num = 0
                else:
                    rand_num = random.randint(0, np.size(temp_arr, 1)-1)
                if np.size(temp_arr, 1) == 0:
                    break
                else:
                    temp_arr = np.delete(temp_arr, rand_num, 1)
        end = False
        if np.size(temp_arr, 1) == 0:
            end = True
            print("--------------------------")
            print("Iteration: " + str(i))
            print("Subset: " + "---")
            print("Accuracy: " + "---")
            print("Pr[Accept]: " + "---")
            print("Random Uniform: " + "---")
            print("Status: Restart")
            break
        perturbed_feature_set = temp_arr
        # Calculating accuracy for original set
        X_train_fold1, X_test_fold1, y_train_fold1, y_test_fold1 = train_test_split(current_feature_set, y, test_size = 0.50, random_state = 1)

        X_train_fold2 = X_test_fold1
        X_test_fold2 = X_train_fold1
        y_train_fold2 = y_test_fold1
        y_test_fold2 = y_train_fold1

        dtree = DecisionTreeClassifier()

        dtree.fit(X_train_fold1, y_train_fold1)
        pred_f1 = dtree.predict(X_test_fold1)
        dtree.fit(X_train_fold2, y_train_fold2)
        pred_f2 = dtree.predict(X_test_fold2)

        predictions = np.concatenate([pred_f1, pred_f2])
        y_test = np.concatenate((y_test_fold1, y_test_fold2))

        org_acc = accuracy_score(y_test, predictions)

        # Calculating accuracy for perturbed set
        X_train_fold1, X_test_fold1, y_train_fold1, y_test_fold1 = train_test_split(perturbed_feature_set, y, test_size = 0.50, random_state = 1)

        X_train_fold2 = X_test_fold1
        X_test_fold2 = X_train_fold1
        y_train_fold2 = y_test_fold1
        y_test_fold2 = y_train_fold1

        dtree = DecisionTreeClassifier()

        dtree.fit(X_train_fold1, y_train_fold1)
        pred_f1 = dtree.predict(X_test_fold1)
        dtree.fit(X_train_fold2, y_train_fold2)
        pred_f2 = dtree.predict(X_test_fold2)

        predictions = np.concatenate([pred_f1, pred_f2])
        y_test = np.concatenate((y_test_fold1, y_test_fold2))

        prime_acc = accuracy_score(y_test, predictions)
        # Compare accuracies
        if prime_acc > org_acc:
            current_feature_set = perturbed_feature_set
            status = "IMPROVED"
            subset = "NEW SUBSET"
            accuracy = prime_acc
            acceptance_probability = "---"
            random_uniform = "---"
        else:
            subset = "ORIGINAL SUBSET"
            accuracy = org_acc
            acceptance_probability = exp((i * -1) * ((org_acc - prime_acc) / org_acc))
            random_uniform = np.random.uniform()
            if random_uniform > acceptance_probability:
                status = "DISCARDED"
            else:
                status = "ACCEPTED"
        print("--------------------------")
        print("Iteration: " + str(i))
        print("Subset: " + subset)
        print("Accuracy: " + str(accuracy))
        print("Pr[Accept]: " + str(acceptance_probability))
        print("Random Uniform: " + str(random_uniform))
        print("Status: " + status)

dtree = DecisionTreeClassifier()

dtree.fit(X_train_fold1, y_train_fold1)
pred_f1 = dtree.predict(X_test_fold1)
dtree.fit(X_train_fold2, y_train_fold2)
pred_f2 = dtree.predict(X_test_fold2)

predictions = np.concatenate([pred_f1, pred_f2])
y_test = np.concatenate((y_test_fold1, y_test_fold2))

print("features:" + ' sepal-length'+' sepal-width'+' petal-length'+' petal-width'+' sepal-length-prime'+' sepal-width-prime'+' petal-length-prime'+' petal-width-prime')
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nAccuracy Score:")
print(accuracy_score(y_test, predictions))
print("\n\n-------------------------------------------------")

print("Part 4: Using the genetic algorithm")
X_transformed = X
X = array[:, 0:4]
y = array[:,4]
feature_names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'sepal-length-prime', 'sepal-width-prime', 'petal-length-prime', 'petal-width-prime']
combined_features_orig = np.column_stack((X, X_transformed))
X_train, X_validation, Y_train, Y_validation = train_test_split(combined_features_orig, y, test_size=0.50, random_state=1, shuffle=True)

def fitness_score(population):
    scores = []
    for chromosome in population:
        dtree.fit(X_train[:,chromosome],Y_train)
        predictions = dtree.predict(X_validation[:,chromosome])
        scores.append(accuracy_score(Y_validation,predictions))
    scores, population = np.array(scores), np.array(population)
    inds = np.argsort(scores)
    return list(scores[inds][::-1]), list(population[inds,:][::-1])

def mutation(pop_after_cross,mutation_rate):
    population_nextgen = []
    for i in range(0,len(pop_after_cross)):
        chromosome = pop_after_cross[i]
        for j in range(len(chromosome)):
            if random.random() < mutation_rate:
                chromosome[j]= not chromosome[j]
        population_nextgen.append(chromosome)
    #print(population_nextgen)
    return population_nextgen

def crossover(pop_after_sel):
    population_nextgen=pop_after_sel
    for i in range(len(pop_after_sel)):
        child=pop_after_sel[i]
        child[3:7]=pop_after_sel[(i+1)%len(pop_after_sel)][3:7]
        population_nextgen.append(child)
    return population_nextgen

def selection(pop_after_fit,n_parents):
    population_nextgen = []
    for i in range(n_parents):
        population_nextgen.append(pop_after_fit[i])
    return population_nextgen

population = []
chromosome = np.ones(8,dtype=np.bool)
chromosome[5:8] = False
population.append(chromosome)
chromosome = np.ones(8,dtype=np.bool)
chromosome[6:8] = False
chromosome[2] = False
population.append(chromosome)
chromosome = np.ones(8,dtype=np.bool)
chromosome[7] = False
chromosome[0] = False
chromosome[3] = False
population.append(chromosome)
chromosome = np.ones(8,dtype=np.bool)
chromosome[0] = False
chromosome[2:3] = False
population.append(chromosome)
chromosome = np.ones(8,dtype=np.bool)
chromosome[1:3] = False
population.append(chromosome)
# for i in range(5):
#     chromosome = np.ones(5,dtype=np.bool)
#     print(chromosome)
#     chromosome[0:i] = False
#     population.append(chromosome)

best_chromo= []
best_score= []
for i in range(50):
    print(population)
    scores, pop_after_fit = fitness_score(population)
    print(scores[:])
    pop_after_sel = selection(pop_after_fit,3)
    pop_after_cross = crossover(pop_after_sel)
    population = mutation(pop_after_cross,0.1)
    best_chromo.append(pop_after_fit[0])
    best_score.append(scores[0])
    print("Gen: ", i)
#
# dtree = DecisionTreeClassifier()
# dtree.fit(X_train[:,best_chromo[-1]],Y_train)
# predictions = dtree.predict(Y_train[:,best_score[-1]])
# print("Accuracy score after genetic algorithm is= "+str(accuracy_score(Y_validation,predictions)))
