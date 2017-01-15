from sklearn.cluster import KMeans
import numpy as np
import pandas, random
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import TfidfVectorizer

if __name__ == "__main__":
    number_of_documents = 1000
    LI_csv = "LI.csv"
    SONAR_csv = "SONAR.csv"
    SONARLE_csv = "SONARLE.csv"


    LI_df, SONAR_df = pandas.read_csv(LI_csv), pandas.read_csv(SONAR_csv)
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

    # Perfomance measure
    overall_accuracy = []

    # Validation method: k-fold
    kf = KFold(len(total_text), n_folds=10, shuffle=False)
    for train, test in kf:

        # Vectorization
        vectorizer = TfidfVectorizer()
        train_text, test_text = total_text[train], total_text[test]

        # Building the train- and testset
        X_train, X_test  = vectorizer.fit_transform(train_text), vectorizer.transform(test_text)
        y_train, y_test= total_labels[train], total_labels[test]

        # Clustering
        true_k = 2
        model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)

        # Fitting the model
        model.fit(X_train)

        # Predicting the test set
        predictlabels = model.predict(X_test)


        # Performance measure for each k-fold iteration
        accuracy = float(np.sum(predictlabels == y_test))
        print "accuracy = " + str(accuracy/len(test_text))
        if accuracy/len(test_text) < 0.4:
            overall_accuracy.append(1-accuracy/len(test_text))
        else:
            overall_accuracy.append(accuracy/len(test_text))


    # Performance measure across all k-fold iterations
    print "Overall accuracy = " + str(np.mean(overall_accuracy))
