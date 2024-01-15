from matplotlib import pyplot as plt
import os
import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import pandas as pd
import pickle
import seaborn as sns

### define variables
max_sift_bow_feature_number = 300 # maximum number of features extracted per image for bow
max_sift_df_feature_number = 2000 # maximum number of features extracted per image for picture histograms
n_clusters = 512 # number of visual words in bag of words
svm = 'linear' # type of svm ('linear' uses one vs rest, None uses standard svc of sklearn with one vs one)
train_dir = os.path.join(os.getcwd(),'dataset','train') # path to trainining set
test_dir = os.path.join(os.getcwd(),'dataset','test') # path to testing set

print('#########################################################################')
print('Start of BOW pipeline - if recalculation of specific saved data is')
print('is desired, please delete this data (place to save is current dir)')
print('params are:')
print(f'\tmax_sift_bow_feature_number = {max_sift_bow_feature_number}')
print(f'\tmax_sift_df_feature_number = {max_sift_df_feature_number}')
print(f'\tn_cluster = {n_clusters}')
print('#########################################################################\n')

# SIFT feature extraction from training set
print('Start SIFT feature extraction')
# create SIFT
sift_bow = cv2.SIFT_create(max_sift_bow_feature_number) # keypoint extractor for bow
sift_df = cv2.SIFT_create(max_sift_df_feature_number) # keypoint extractor for histograms

try:
    features = np.load('sift_features.npy')
    corresponding_images = np.load('corresponding_images.npy')
    print('features loaded from disc')
except:
    print('Features have to be calculated')
    # extract keypoints and corresping sift descriptors, then add descriptors to
    # features list and safe indices of keypoints per picture to calc inverse
    # document frequencies
    features = np.empty((0,128))
    corresponding_images = []
    i = 0
    for subdir, dirs, files in os.walk(train_dir):
        print(subdir)
        folder_features = np.empty((0,128))
        folder_corrs = []
        for jpg in files:
            image = cv2.imread(os.path.join(subdir,jpg))
            kp,descriptors = sift_bow.detectAndCompute(image,None)
            # correspondencies are in the format (<filename>, <start_index>, <end_index>) where start_index is inclusive, while end_index is exclusive
            folder_corrs = folder_corrs + [(jpg, i, i + len(kp))]
            i += len(kp)
            folder_features = np.concatenate((folder_features, descriptors), axis=0)
        features = np.concatenate((features, folder_features))
        corresponding_images = corresponding_images + folder_corrs
    np.save('sift_features', features)
    np.save('corresponding_images', np.asarray(corresponding_images))

print(f'Found a total of {len(features):,} features')
print('SIFT feature extraction done ############################################\n')

print('Start kmeans')
# put all features into n_clusters bins which represent the visual words
# standatize features first
kmeans_scaler = StandardScaler()
features = kmeans_scaler.fit_transform(features)
try:
    with open("kmeans", 'rb') as f:
        kmeans = pickle.load(f)
    print('kmeans loaded from disk')
except:
    print('kmeans has to be calculated')
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", verbose=1, max_iter=1000).fit(features)
    with open("kmeans", 'wb') as f:
        pickle.dump(kmeans, f, protocol=5)

# extryct labels for future use
labels = kmeans.labels_
print('kmeans done #############################################################\n')

print('Start generating training histograms')
# get inverse document frequencies
inverted_file_count = [0 for i in range(n_clusters)]
for _, i0, i1 in corresponding_images:
    image_words = np.unique(labels[int(i0):int(i1)])
    for i in image_words:
        inverted_file_count[i] += 1

N = len(corresponding_images)
inverse_document_freq = [N / inverted_file_count[i] for i in range(n_clusters)]
inverse_document_freq = np.log2(inverse_document_freq)

# get the histograms of the training images
try:
    with open("df_train", 'rb') as f:
        df_train = pickle.load(f)
    print('df_test loaded from disc')
except:
    print('Calclulating df_train')
    train_set = []
    inverted_file_count = [0 for i in range(n_clusters)]
    for subdir, dirs, files in os.walk(train_dir):
        jpg_class = subdir.split('/')[-1] # extract class to which the following pics belong
        print(subdir)
        subdir_ts = []
        for jpg in files:
            image = cv2.imread(os.path.join(subdir,jpg))
            kp,descriptors = sift_df.detectAndCompute(image,None)

            # classify train data (which cluster) (f_predict of clustering)
            descriptors = kmeans_scaler.transform(descriptors)
            home_clusters = kmeans.predict(list(descriptors))

            image_words = np.unique(home_clusters)
            for i in image_words:
                inverted_file_count[i] += 1

            # generate histogram(s)
            hist = [np.count_nonzero(home_clusters == i) / len(kp) for i in range(n_clusters)] # word frequencies
            hist = np.multiply(hist, inverse_document_freq) # multiplied with logarithm of inverse document frequency
            hist = hist / np.sum(hist) # normalize histogram
            subdir_ts = subdir_ts + [list(hist) + [jpg_class]]
        train_set = train_set + subdir_ts

    df_train = pd.DataFrame(train_set, columns = [f'cluster{i}' for i in range(n_clusters)] + ["category"])
    with open("df_train", 'wb') as f:
        pickle.dump(df_train, f, protocol=5)
# print('df_train information')
# df_train.info()
# print(df_train.head())
print('Generating training set histograms done #################################\n')

# prepocess dataset for use in classifiers
print('Start ml preprocessing')
# create working copy
df_train_work = df_train.copy()

# encode labels
label_encoder = LabelEncoder()
df_train_work["labeled_category"] = label_encoder.fit_transform(df_train_work["category"])
categories = label_encoder.classes_ # extract all different categories of pics to be classified
X_train = df_train_work[[f'cluster{i}' for i in range(n_clusters)]]
y_train = df_train_work["labeled_category"]

# scale values
scaler = MinMaxScaler() # use MinMaxScaler to preserve 0 entries
# X_train = scaler.fit_transform(X_train)
print('ml preprocessing done ###################################################\n')


# train knearest neighbor classifier (with k = 1 so its basically the closest histogram)
print('Start training knearest neighbor classifier for training set')
try:
    with open("neighbor_clf", 'rb') as f:
        neighbor_clf = pickle.load(f)
    print('neighbor_clf loaded from disc')
except:
    print('Neighbor clf has to be calculated')
    neighbor_clf = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)
    with open("neighbor_clf", 'wb') as f:
        pickle.dump(neighbor_clf, f, protocol=5)
print('Training of knearest neighbor for training set done #####################\n')

# train multiclass svm
print('Start training svm classifier for training set')
try:
    with open("svm_clf", 'rb') as f:
        svm_clf = pickle.load(f)
    print('svm_clf loaded from disc')
except:
    print('SVM clf has to be calculated')
    if svm == 'linear':
        svc = LinearSVC(dual='auto', max_iter=5000) # linearSVC uses one vs rest as multiclass
        param_grid = dict(C = [2**p for p in range(-5, 16)])
    else:
        svc = SVC() # gaussian svc doesn't use one vs rest
        param_grid = dict(gamma = [2**p for p in range(-15, 4)],
                          C = [2**p for p in range(-5, 16)])
    svm_clf = GridSearchCV(svc, param_grid, verbose=3, n_jobs=-1)
    svm_clf.fit(X_train, y_train)
    print(f"Best params are {svm_clf.best_params_}")
    with open("svm_clf", 'wb') as f:
        pickle.dump(svm_clf, f, protocol=5)
print('Training of svm for training set done ###################################\n')


################################################################################
# load test data
print('#########################################################################')
print('Starting with testset')
print('#########################################################################\n')

print('Start generating test histograms')
try:
    with open("df_test", 'rb') as f:
        df_test = pickle.load(f)
        print('df_test loaded from disc')
except:
    print('Calculating df_test')
    test_set = []
    for subdir, dirs, files in os.walk(test_dir):
        jpg_class = subdir.split('/')[-1] # extract class to which the following pics belong
        print(subdir)
        subdir_ts = []
        for jpg in files:
            image = cv2.imread(os.path.join(subdir,jpg))
            kp,descriptors = sift_df.detectAndCompute(image,None)

            # classify test data (which cluster) (f_predict of clustering)
            descriptors = kmeans_scaler.transform(descriptors)
            home_clusters = kmeans.predict(list(descriptors))

            # generate histogram(s)
            hist = [np.count_nonzero(home_clusters == i) / len(kp) for i in range(n_clusters)] # word frequencies
            hist = np.multiply(hist, inverse_document_freq) # multiplied with logarithm of inverse document frequency
            hist = hist / np.sum(hist) # normalize histogram
            subdir_ts = subdir_ts + [list(hist) + [jpg_class]]
        test_set = test_set + subdir_ts

    df_test = pd.DataFrame(test_set, columns = [f'cluster{i}' for i in range(n_clusters)] + ["category"])
    with open("df_test", 'wb') as f:
        pickle.dump(df_test, f, protocol=5)

# print('df_test information')
# df_test.info()
# print(df_test.head())
print('Generating test set histograms done #####################################\n')

# prepare test data for ml algs
print('Start preprocessing of test data')
# create working copy
df_test_work = df_test.copy()

# encode labels
df_test_work["labeled_category"] = label_encoder.transform(df_test_work["category"])
X_test = df_test_work[[f'cluster{i}' for i in range(n_clusters)]]
y_test = df_test_work["labeled_category"]

# scale values
# X_test = scaler.transform(X_test)

print('Preprocessing test data done ############################################\n')

# classify test data with nearest neighbor
y_classified_neighbor_labeled = neighbor_clf.predict(X_test)
y_classified_neighbor = label_encoder.inverse_transform(y_classified_neighbor_labeled)

# classify test data with svm
y_classified_svm_labeled = svm_clf.predict(X_test)
y_classified_svm = label_encoder.inverse_transform(y_classified_svm_labeled)

# calculate confusion matrix for nearst neighbour and svm
conf_mat_neighbor = confusion_matrix(y_test, y_classified_neighbor_labeled, normalize = 'true')
conf_mat_svm = confusion_matrix(y_test, y_classified_svm_labeled, normalize = 'true')

plt.figure(figsize=(8,6))
sns.heatmap(conf_mat_neighbor, vmin=0, vmax=1, annot=True, fmt='.1%', annot_kws={'size': 6})
plt.title("BoW with kNN Classifier")
plt.xlabel('predicted class')
plt.ylabel('image class')
plt.savefig(f'confmat_knn_{n_clusters}.pdf')
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(conf_mat_svm, vmin=0, vmax=1, annot=True, fmt='.1%', annot_kws={'size': 6})
plt.title("BoW with Support Vector Classifier")
plt.xlabel('predicted class')
plt.ylabel('image class')
plt.savefig(f'confmat_svm_{n_clusters}.pdf')
plt.show()

print('done ####################################################################\n')
