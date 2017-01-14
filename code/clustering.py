from sklearn.cluster import KMeans
import numpy as np
import pandas, random
from sklearn.feature_extraction.text import TfidfVectorizer

number_of_documents = 1000
LI_csv = "LI.csv"
SONAR_csv = "SONAR.csv"
SONARLE_csv = "SONARLE.csv"

LI_df, SONAR_df = pandas.read_csv(LI_csv), pandas.read_csv(SONARLE_csv)
LI_text, SONAR_text = np.asarray(LI_df['Text']), np.asarray(SONAR_df['Text'])
LI_labels, SONAR_labels = [0] * number_of_documents, [1] * number_of_documents

total_text = np.concatenate((LI_text, SONAR_text),axis=0)
total_labels = LI_labels + SONAR_labels

temp = list(zip(total_text, total_labels))
random.shuffle(temp)
total_text, total_labels = zip(*temp)
total_text, total_labels = np.asarray(total_text), np.asarray(total_labels)
test_text = total_text[900:1000]
test_labels = total_labels[900:1000]
# random.shuffle(test_labels)
train_text = total_text[:900]
train_labels = total_labels[:900]
with open('C:\Users\Tanja\Documents\Project paper\Text-mining-final-project\code\stopwords.txt', 'r') as f:
    stopwords = f.readlines()
vectorizer = TfidfVectorizer(stop_words=stopwords)
X = vectorizer.fit_transform(train_text)
Y = vectorizer.transform(test_text)

true_k = 2
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)
predictlabels = model.predict(Y)
print "accuracy = " + str(float(np.sum(predictlabels == test_labels))/100)

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print "Cluster %d:" % i,
    for ind in order_centroids[i, :50]:
        print ' %s' % terms[ind],
    print