import pandas as pd
import numpy as np
from math import sqrt, pi, exp
import matplotlib.pyplot as plt

df = pd.read_csv('iris.csv', header=None,
                      names=['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])
df['class'] = df['class'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

#cast as int
df['class'] = df['class'].astype(int)

#shuffle data for randomization
df = df.sample(frac=1)


#Splitting
train_set = df.iloc[40:150]
test_set = df.iloc[40:]

#
class Gaussian:
    def __init__(self, mean, stdev):
        self.mean = mean
        self.stdev = stdev

    def calculate_probability(self, x):
        return (1 / (self.stdev * np.sqrt(2 * np.pi))) * np.exp(-((x - self.mean) ** 2)
                                                              / (2 * self.stdev ** 2))
"""
implements a class with a function that computes the probability of a data point belonging to each class.
"""

class NaiveBayes:
    def __init__(self, df):
        self.df = df

    def calculate_class_probabilities(self, row):
        probabilities = {}
        for class_name, class_df in self.df.groupby('class'):
            probabilities[class_name] = 1
            for i in range(len(class_df.columns) - 1):
                mean = class_df[class_df.columns[i]].mean()
                stdev = class_df[class_df.columns[i]].std()
                probabilities[class_name] *= Gaussian(mean, stdev).calculate_probability(row[i])
        return probabilities

    def predict(self, row):
        probabilities = self.calculate_class_probabilities(row)
        best_label, best_prob = None, -1
        for class_name, probability in probabilities.items():
            if best_label is None or probability > best_prob:
                best_prob = probability
                best_label = class_name
        return best_label

    def get_predictions(self, test_set):
        predictions = []
        for index, row in test_set.iterrows():
            predictions.append(self.predict(row))
        return predictions

    def get_accuracy(self, test_set):
        correct_predictions = 0
        predictions = self.get_predictions(test_set)
        for i in range(len(predictions)):
            if predictions[i] == test_set.iloc[i][-1]:
                correct_predictions += 1
        return correct_predictions / len(predictions)

naive_bayes = NaiveBayes(train_set)
print('Accuracy of train set: {:.2f}%' .format(naive_bayes.get_accuracy(train_set) * 100))
print('Accuracy of test set: {:.2f}%' .format(naive_bayes.get_accuracy(test_set) * 100))


def bootstrap_aggregation(df, m):
    set_list = []
    for i in range(m):
        set_list.append(train_set.sample(frac=1, replace=True))

    return set_list

def get_accuracy(df, m):
    set_list = bootstrap_aggregation(df, m)
    accuracies = []

    for i in range(m):
        naive_bayes = NaiveBayes(set_list[i])
        accuracies.append(naive_bayes.get_accuracy(test_set))
    return accuracies

"""
get the mean and standard deviations of the accuracies
"""
def get_mean_and_std(accuracies):
    mean = np.mean(accuracies)
    std = np.std(accuracies)
    return mean, std

accuracies = get_accuracy(df, 8)
mean, standard_deviation = get_mean_and_std(accuracies)
print('The mean of the accuracy: {:.2f}%'.format(mean *100))
print('The standard deviation of the accuracy: {:.2f}%'.format(standard_deviation *100))

plt.plot(accuracies)
plt.ylabel('Accuracy')
plt.xlabel('Dataset')
plt.show()

"""
train the model on each dataset and get the accuracy.
"""

df_list = bootstrap_aggregation(df, 5)
for i in range(len(df_list)):
    naive_bayes = NaiveBayes(df_list[i])
    print('Accuracy of bagged set #',i+1,': {:.2f}%' .format(naive_bayes.get_accuracy(test_set) * 100))


class MajorityVote:
    def __init__(self, df_list):
        self.df_list = df_list

    def get_predictions(self, test_set):
        predictions = []
        for index, row in test_set.iterrows():
            class_0_count = 0
            class_1_count = 0
            class_2_count = 0
            for i in range(len(self.df_list)):
                prediction = NaiveBayes(self.df_list[i]).predict(row)
                if prediction == 0:
                    class_0_count += 1
                elif prediction == 1:
                    class_1_count += 1
                else:
                    class_2_count += 1
            if class_0_count > class_1_count and class_0_count > class_2_count:
                predictions.append(0)
            elif class_1_count > class_0_count and class_1_count > class_2_count:
                predictions.append(1)
            else:
                predictions.append(2)
        return predictions

    def get_accuracy(self, test_set):
        correct_predictions = 0
        predictions = self.get_predictions(test_set)
        for i in range(len(predictions)):
            if predictions[i] == test_set.iloc[i][-1]:
                correct_predictions += 1
        return correct_predictions / len(predictions)


"""
train the model on each dataset and get the accuracy.
"""

df_list = bootstrap_aggregation(df, 5)
majority_vote = MajorityVote(df_list)
print('The accuracy of the majority vote: {:.2f}%' .format(majority_vote.get_accuracy(test_set)* 100))










