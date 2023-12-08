import re
import pandas as pd
import pyttsx3
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import csv
import numpy as np 
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def healthcare_chatbot(training_path, testing_path):
    def getSeverityDict():
        severity_dict = {}
        with open("C:/Users/sahil/OneDrive/Desktop/Major Project/data set 3/master data/Symptom_severity.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                _diction = {row[0]: int(row[1])}
                severity_dict.update(_diction)
        return severity_dict

    def getDescription():
        description_list = {}
        with open("C:/Users/sahil/OneDrive/Desktop/Major Project/data set 3/master data/symptom_Description.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                _description = {row[0]: row[1]}
                description_list.update(_description)
        return description_list

    def getprecautionDict():
        precaution_dict = {}
        with open("C:/Users/sahil/OneDrive/Desktop/Major Project/data set 3/master data/symptom_precaution.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                _prec = {row[0]: [row[1], row[2], row[3], row[4]]}
                precaution_dict.update(_prec)
        return precaution_dict

    def sec_predict(symptoms_exp, clf):
        df = pd.read_csv("C:/Users/sahil/OneDrive/Desktop/Major Project/data set 3/Training.csv")
        X = df.iloc[:, :-1]
        y = df['prognosis']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
        rf_clf = DecisionTreeClassifier()
        rf_clf.fit(X_train, y_train)

        symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
        input_vector = np.zeros(len(symptoms_dict))
        for item in symptoms_exp:
            input_vector[[symptoms_dict[item]]] = 1

        return rf_clf.predict([input_vector])

    def print_disease(node, le):
        node = node[0]
        val = node.nonzero()
        disease = le.inverse_transform(val[0])
        return list(map(lambda x: x.strip(), list(disease)))

    def tree_to_code(tree, feature_names, clf, cols):
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]

        chk_dis = ",".join(feature_names).split(",")
        symptoms_present = []

        while True:
            print("\nEnter the symptom you are experiencing  \t\t", end="->")
            disease_input = input("")
            conf, cnf_dis = check_pattern(chk_dis, disease_input)
            if conf == 1:
                print("searches related to input: ")
                for num, it in enumerate(cnf_dis):
                    print(num, ")", it)
                if num != 0:
                    print(f"Select the one you meant (0 - {num}):  ", end="")
                    conf_inp = int(input(""))
                else:
                    conf_inp = 0

                disease_input = cnf_dis[conf_inp]
                break
            else:
                print("Enter valid symptom.")

        while True:
            try:
                num_days = int(input("Okay. From how many days ? : "))
                break
            except:
                print("Enter valid input.")

        def recurse(node, depth):
            indent = "  " * depth
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]

                if name == disease_input:
                    val = 1
                else:
                    val = 0
                if val <= threshold:
                    recurse(tree_.children_left[node], depth + 1)
                else:
                    symptoms_present.append(name)
                    recurse(tree_.children_right[node], depth + 1)
            else:
                present_disease = print_disease(tree_.value[node], le)
                red_cols = reduced_data.columns
                symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
                print("Are you experiencing any ")
                symptoms_exp = []
                for syms in list(symptoms_given):
                    inp = ""
                    print(syms, "? : ", end='')
                    while True:
                        inp = input("")
                        if inp == "yes" or inp == "no":
                            break
                        else:
                            print("provide proper answers i.e. (yes/no) : ", end="")
                    if inp == "yes":
                        symptoms_exp.append(syms)

                second_prediction = sec_predict(symptoms_exp, clf)
                calc_condition(symptoms_exp, num_days)
                if present_disease[0] == second_prediction[0]:
                    print("You may have ", present_disease[0])
                    print(description_list[present_disease[0]])

                else:
                    print("You may have ", present_disease[0], "or ", second_prediction[0])
                    print(description_list[present_disease[0]])
                    print(description_list[second_prediction[0]])

                precution_list = precaution_dict[present_disease[0]]
                print("Take following measures : ")
                for i, j in enumerate(precution_list):
                    print(i + 1, ")", j)

        recurse(0, 1)

    def check_pattern(dis_list, inp):
        pred_list = []
        inp = inp.replace(' ', '_')
        patt = f"{inp}"
        regexp = re.compile(patt)
        pred_list = [item for item in dis_list if regexp.search(item)]
        if len(pred_list) > 0:
            return 1, pred_list
        else:
            return 0, []

    def calc_condition(exp, days):
        sum = 0
        for item in exp:
            sum = sum + severity_dict[item]
        if (sum * days) / (len(exp) + 1) > 13:
            print("You should take the consultation from the doctor.")
        else:
            print("It might not be that bad but you should take precautions.")

    def getInfo():
        print("-----------------------------------HealthCare ChatBot-----------------------------------")
        print("\nYour Name? \t\t\t\t", end="->")
        name = input("")
        print("Hello, ", name)

    # Main script starts here
    training = pd.read_csv(training_path)
    testing = pd.read_csv(testing_path)
    cols = training.columns
    cols = cols[:-1]
    x = training[cols]
    y = training['prognosis']
    y1 = y

    reduced_data = training.groupby(training['prognosis']).max()

    # Mapping strings to numbers
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    testx = testing[cols]
    testy = testing['prognosis']
    testy = le.transform(testy)

    clf1 = DecisionTreeClassifier()
    clf = clf1.fit(x_train, y_train)
    scores = cross_val_score(clf, x_test, y_test, cv=3)
    print("Decision Tree Cross Validation Score: ", scores.mean())

    model = SVC()
    model.fit(x_train, y_train)
    print("SVM Score: ", model.score(x_test, y_test))

    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    severity_dict = getSeverityDict()
    description_list = getDescription()
    precaution_dict = getprecautionDict()

    getInfo()
    tree_to_code(clf, cols, clf, cols)
    print("----------------------------------------------------------------------------------------")

healthcare_chatbot("C:/Users/sahil/OneDrive/Desktop/Major Project/data set 3/Training.csv",
                   "C:/Users/sahil/OneDrive/Desktop/Major Project/data set 3/Testing.csv")
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def find_matching_pincodes(excel_file_path, input_pincode):
    # Load the Excel file containing the pincode data
    data = pd.read_excel(excel_file_path)

    # Extract the pincode column
    pincode_column = data['PINCODE'].astype(str)

    # Create a TfidfVectorizer to convert pincode data into numerical vectors
    vectorizer = TfidfVectorizer()
    pincode_vectors = vectorizer.fit_transform(pincode_column)

    # Convert the input pincode into a numerical vector
    input_vector = vectorizer.transform([input_pincode])

    # Calculate cosine similarity between the input and all pincode vectors
    cosine_similarities = cosine_similarity(input_vector, pincode_vectors)

    # Find indices where cosine similarity is 1 (exact matches)
    matching_indices = [idx for idx, similarity in enumerate(cosine_similarities[0]) if similarity == 1]

    # Return matching pincode entries
    matching_entries = []
    for idx in matching_indices:
        entry = {
            "Pincode": data.iloc[idx]['PINCODE'],
            "Location": data.iloc[idx]['HOSPITAL NAME']
        }
        matching_entries.append(entry)

    return matching_entries

# Example usage:
excel_file_path = "C:/Users/sahil/OneDrive/Desktop/Major Project/data set 3/DELHI PPN HOSPITALS.xlsx"
input_pincode = input("Enter a pincode: ")
result = find_matching_pincodes(excel_file_path, input_pincode)

if result:
    print("Matching pincode entries:")
    for entry in result:
        print(f"Pincode: {entry['Pincode']}, Location: {entry['Location']}")
else:
    print("No matching pincode entries found.")


