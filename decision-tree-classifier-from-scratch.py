import math
import random
import matplotlib.pyplot as plt

with open('/Users/peterlewis/Desktop/VSCODE/PYTHON/Intelligent Systems/Assignment2/Task1/car.csv', 'r') as file:
    lines = file.readlines()

data = [line.strip().split(',') for line in lines]

# Function to split dataset into training and testing sets.
def train_test_split(data, test_size=0.2):
    random.shuffle(data)
    split_idx = int(len(data) * (1 - test_size))
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    return train_data, test_data

train_data, test_data = train_test_split(data)

# Entropy calculation function
def entropy(data):
    labels = [row[-1] for row in data]
    label_counts = {label: labels.count(label) for label in set(labels)}
    entropy_val = -sum((count / len(data)) * math.log2(count / len(data)) for count in label_counts.values())
    return entropy_val

# Information gain calculation function
def information_gain(data, attribute_idx):
    total_entropy = entropy(data)
    values, counts = zip(*[(row[attribute_idx], 1) for row in data])
    values, counts = list(values), list(counts)
    weighted_entropy = sum((counts[i] / len(data)) * entropy([row for row in data if row[attribute_idx] == values[i]]) for i in range(len(values)))
    information_gain_val = total_entropy - weighted_entropy
    return information_gain_val

# Function to select best attribute based on information gain
def select_best_attribute(data):
    num_attributes = len(data[0]) - 1
    information_gains = {i: information_gain(data, i) for i in range(num_attributes)}
    best_attribute = max(information_gains, key=information_gains.get)
    return best_attribute

# Decision tree node class
class Node:
    def __init__(self, attribute):
        self.attribute = attribute
        self.children = {}
        self.prediction = None

# Function to build decision tree
def build_tree(data, attributes):
    if len(set([row[-1] for row in data])) == 1:
        leaf = Node(None)
        leaf.prediction = data[0][-1]
        return leaf
    if len(attributes) == 0:
        leaf = Node(None)
        leaf.prediction = max(set([row[-1] for row in data]), key=[row[-1] for row in data].count)
        return leaf
    
    best_attribute = select_best_attribute(data)
    attributes.remove(best_attribute)
    
    tree = Node(best_attribute)
    for value in set([row[best_attribute] for row in data]):
        sub_data = [row for row in data if row[best_attribute] == value]
        if len(sub_data) == 0:
            leaf = Node(None)
            leaf.prediction = max(set([row[-1] for row in data]), key=[row[-1] for row in data].count)
            tree.children[value] = leaf
        else:
            tree.children[value] = build_tree(sub_data, attributes.copy())
    
    return tree

# Function to make predictions
def predict(tree, example):
    if tree.prediction is not None:
        return tree.prediction
    attribute = tree.attribute
    if example[attribute] not in tree.children:
        return random.choice(list(tree.children.values())).prediction
    return predict(tree.children[example[attribute]], example)

attributes = list(range(len(train_data[0]) - 1))
tree = build_tree(train_data, attributes.copy())

# Evaluate the decision tree
def evaluate(tree, test_data):
    correct = 0
    for example in test_data:
        prediction = predict(tree, example)
        if prediction == example[-1]:
            correct += 1
    accuracy = correct / len(test_data)
    return accuracy


# Confusion matrix
confusion_matrix = {
    'unacc': {'unacc': 0, 'acc': 0, 'good': 0, 'vgood': 0},
    'acc': {'unacc': 0, 'acc': 0, 'good': 0, 'vgood': 0},
    'good': {'unacc': 0, 'acc': 0, 'good': 0, 'vgood': 0},
    'vgood': {'unacc': 0, 'acc': 0, 'good': 0, 'vgood': 0}
}

for example in test_data:
    prediction = predict(tree, example)
    actual = example[-1]
    confusion_matrix[actual][prediction] += 1
    
print("Confusion Matrix:")
for actual_class, predicted_counts in confusion_matrix.items():
    print(f"Actual {actual_class}:", end=" ")
    for predicted_class, count in predicted_counts.items():
        print(f"{count:5d}", end=" ")
    print()

# Calculate precision, recall and F1-score
precision = {}
recall = {}
f1_score = {}
for cls in confusion_matrix:
    true_positive = confusion_matrix[cls][cls]
    false_positive = sum(confusion_matrix[pred_cls][cls] for pred_cls in confusion_matrix if pred_cls != cls)
    false_negative = sum(confusion_matrix[cls][pred_cls] for pred_cls in confusion_matrix if pred_cls != cls)
    precision[cls] = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0
    recall[cls] = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0
    f1_score[cls] = 2 * (precision[cls] * recall[cls]) / (precision[cls] + recall[cls]) if (precision[cls] + recall[cls]) != 0 else 0

# Macro-average
macro_precision = sum(precision[cls] for cls in precision) / len(precision)
macro_recall = sum(recall[cls] for cls in recall) / len(recall)
macro_f1_score = sum(f1_score[cls] for cls in f1_score) / len(f1_score)

# Weighted-average
weighted_precision = sum(confusion_matrix[cls][cls] * precision[cls] for cls in precision) / sum(sum(row.values()) for row in confusion_matrix.values())
weighted_recall = sum(confusion_matrix[cls][cls] * recall[cls] for cls in recall) / sum(sum(row.values()) for row in confusion_matrix.values())
weighted_f1_score = sum(confusion_matrix[cls][cls] * f1_score[cls] for cls in f1_score) / sum(sum(row.values()) for row in confusion_matrix.values())

print("Precision, Recall, and F1-score for each class:")
for cls in precision:
    print(f"Class: {cls}")
    print(f"Precision: {precision[cls]:.2f}")
    print(f"Recall: {recall[cls]:.2f}")
    print(f"F1-score: {f1_score[cls]:.2f}\n")

print("Macro-average:")
print(f"Precision: {macro_precision:.2f}")
print(f"Recall: {macro_recall:.2f}")
print(f"F1-score: {macro_f1_score:.2f}\n")

print("Weighted-average:")
print(f"Precision: {weighted_precision:.2f}")
print(f"Recall: {weighted_recall:.2f}")
print(f"F1-score: {weighted_f1_score:.2f}")

print("Size of training set:",len(train_data))
print("Size of testing set:",len(test_data))

accuracy = evaluate(tree, test_data)
print("Total accuracy:", accuracy)

num_samples = range(1, len(train_data) + 1, 100)
accuracies = []
for i in num_samples:
    tree = build_tree(train_data[:i], attributes.copy())
    accuracy = evaluate(tree, test_data)
    accuracies.append(accuracy)

plt.plot(num_samples, accuracies, marker='o')
plt.title('Learning Curve')
plt.xlabel('Number of Training Examples')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()
