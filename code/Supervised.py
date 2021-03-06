import numpy as np
import pandas, re, sklearn, random
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import glob

# This fucntion takes n paths to the documents and returns a csv file of the bag of word representation of all of the documents.
def write_to_xls(csv,path,n):
    stopwords = []
    with open("stopwords.txt", 'r') as fl:
        lines = fl.readlines()
        for line in lines:
            line = line[:-1]
            stopwords.append(line)
    paths = glob.glob(path)[:n]
    noxml = ""
    documents = []
    for p in paths:
        with open(p, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line_no_stop = []
                # Remove xml tags and their contents
                line = re.sub('<[^>]*>','', line)
                # Remove numerals
                line = re.sub('\d','',line)
                # Remove punctuation
                line = re.sub(r'[^\w\s]','',line)
                line = line.strip()
                for word in line.split():
                    if word not in stopwords:
                        line_no_stop.append(word)
                line = " ".join(line_no_stop)
                if line != "":
                    noxml += line + " "
            documents.append(noxml)
            noxml = ""
    df = pandas.DataFrame(documents,columns=["Text"])
    df.to_csv(csv)


# This function return the average word length of a list with multiple documents represented as strings
def average_len_words(lst):
    all_lens = []
    for document in lst:
        all_lens.append(len(document.split()))
    return np.mean(all_lens)

if __name__ == "__main__":
    number_of_documents = 1000
    LIpath = "C:\Users\Tanja\Documents\Text mining data\LI\data\*"
    SONARpath = "C:\Users\Tanja\Documents\Text mining data\NOS\SONAR500\DATA\WR-P-P-G_newspapers\*"
    SONARLEpath = "C:\Users\Tanja\Documents\Text mining data\NOS\SONAR500\DATA\WR-P-P-F_legal_texts\*"
    LI_csv = "LI.csv"
    SONAR_csv = "SONAR.csv"
    SONARLE_csv = "SONARLE.csv"

    # _____________________________________________________________
    # ONLY NECESSARY IF YOU WANT TO CHANGE THE CSV FILE

    # write_to_xls(SONAR_csv, SONARpath, number_of_documents)
    # write_to_xls(LI_csv, LIpath, number_of_documents)
    # write_to_xls(SONARLE_csv, SONARLEpath, number_of_documents)
    # _____________________________________________________________


    LI_df, SONAR_df = pandas.read_csv(LI_csv), pandas.read_csv(SONARLE_csv)
    LI_text, SONAR_text = np.asarray(LI_df['Text']), np.asarray(SONAR_df['Text'])
    LI_labels, SONAR_labels = [0] * number_of_documents, [1] * number_of_documents

    # Text and labels for classification
    total_text = np.concatenate((LI_text, SONAR_text),axis=0)
    total_labels = LI_labels + SONAR_labels

    # Shuffling the lists
    temp = list(zip(total_text, total_labels))
    random.shuffle(temp)
    total_text, total_labels = zip(*temp)
    total_text, total_labels = np.asarray(total_text), np.asarray(total_labels)

    # Evaluation measures
    f1_score_weighted_list = []
    recall_list = []
    precision_list = []
    accuracy_list = []
    total_predictions = []

    # Validation method: k-fold
    kf = KFold(len(total_text), n_folds=10, shuffle=False)
    for train, test in kf:

        # Vectorization
        print "Vectorizing"
        v = TfidfVectorizer()

        # Building the train- and testset
        train_text, test_text = total_text[train], total_text[test]
        X_train, X_test  = v.fit_transform(train_text), v.transform(test_text)
        y_train, y_test= total_labels[train], total_labels[test]

        # The classifier:
        clf = MultinomialNB()

        # Fitting the model
        print "Fitting"
        clf.fit(X_train, y_train)

        # Predicting the test set
        predictions = clf.predict(X_test)

        # Performance measures for each k-fold iteration
        accuracy = metrics.accuracy_score(y_test, predictions)
        accuracy_list.append(accuracy)
        f1_score_weighted = metrics.f1_score(y_test, predictions)
        f1_score_weighted_list.append(f1_score_weighted)
        recall = metrics.recall_score(y_test, predictions, pos_label=0)
        recall_list.append(recall)
        precision = sklearn.metrics.precision_score(y_test, predictions, pos_label=0)
        precision_list.append(precision)

        print metrics.classification_report(y_test, predictions, target_names=["Legal", "Non-legal"])
        total_predictions = total_predictions + list(predictions)

    # Performance measures across all k-fold iterations
    overall_accuracy = np.mean(accuracy_list)
    overall_f1_weighted = np.mean(f1_score_weighted_list)
    overall_recall = np.mean(recall_list)
    overall_precision = np.mean(precision_list)

    print "Accuracy=", overall_accuracy, "\n", "Weighted f1=", overall_f1_weighted, "\n", "Recall=", overall_recall, "\n", "Precision=", overall_precision


