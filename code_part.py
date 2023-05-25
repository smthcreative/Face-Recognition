import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from numpy import linalg as LA
# import mglearn

def read_file(root):
    '''Reading files'''
    root_1 = "archive/s1/1.pgm"
    # root = 'archive'
    # root = 'D:/UCU/Now/Linear Algebra/Face recognition/Training'
    faces = {}
    dir_list = os.listdir(root)
    # dir_list.remove('README')
    # print(dir_list)
    for current_dir in dir_list:
        cur_root = root + '/' + current_dir
        cur_dir = os.listdir(cur_root)
        # print(cur_dir)
        for file in cur_dir:
            final_root = cur_root + "/" + file
            # print(final_root)
            faces[final_root] = cv2.imread(final_root, 0)
    return faces

faces = read_file('D:/UCU/Now/Linear Algebra/Face recognition/Training')
test_faces = read_file('D:/UCU/Now/Linear Algebra/Face recognition/Testing')


''''''
# PCA

# Step 1
# average for now (rows)columns in the matrix
def column_average(matrix: np):
  return matrix.mean(axis=0)

# Step 2
# # mean centering the data(by row)
# def mean_centering_row(matrix: np, average_lst: np):
#   return (matrix.transpose() - average_lst).transpose()
def mean_centering(matrix: np, average_lst: np):
  return (matrix - average_lst)# / matrix.std(axis = 0)

# Step 3
# calculating the covariance matrix
def covariance_matrix(matrix: np):
  return np.dot(matrix.transpose(), matrix)
  # return np.cov(matrix, ddof = 0, rowvar = False)

# Step 4
# eigen decomposition on the covariance matrix to get the eigenvalues and eigenvectors.
def eigen_decomposition(matrix: np):
  return LA.eig(matrix)

# Step 5
# Sort eigenvectors by |eigenvalues|, largest to smallest.
def sorting_eig(eigen_values: np, eigen_vectors: np):
  # eigen_values = abs(eigen_values) # ????
  idx = eigen_values.argsort()[::-1]
  eigen_values = eigen_values[idx]
  eigen_vectors = eigen_vectors[:,idx]
  return eigen_values, eigen_vectors

# Step 6
# Take the top N eigenvectors with the largest corresponding eigenvalue magnitude.
def n_magnitude(eigen_values: np, eigen_vectors: np, n: int):
  n_eigen_values = np.array([eigen_values[i] for i in range(n)])
  tr_eigen_vectors = eigen_vectors.transpose()
  n_eigen_vectors = np.array([tr_eigen_vectors[i] for i in range(n)])
  return n_eigen_values, n_eigen_vectors.transpose()

# Step 7
# Transform the input data by projecting (i.e., taking the dot product) it onto the
# space created by the top N eigenvectors — these eigenvectors are called our eigenfaces.
def projecting(input_data: np, eigen_vectors: np):
  return np.dot(input_data, eigen_vectors)

A = np.array([[3, 5, 1, 4, 4],
              [6, 4, 2, 8, 2],
              [9, 6, 6, 12, 5]])
# print(column_average(A))
# print(mean_centering(A, column_average(A)))

# input_data = np.array([[3, 2, 1],
#                       [6, 4, 2],
#                       [9, 6, 6]])


# print(column_average(A))
def PcA(A,n):
    centered_A = mean_centering(A, column_average(A))
    print(1)
    cov_matrix = covariance_matrix(centered_A)
    print(1)
    # print(cov_matrix)
    # print(centered_A)
    eigen_values, eigen_vectors = eigen_decomposition(cov_matrix)
    print(1)
    # print(eigen_vectors)
    eigen_values, eigen_vectors = sorting_eig(eigen_values, eigen_vectors)
    print(1)
    n_eigen_values, n_eigen_vectors = n_magnitude(eigen_values, eigen_vectors, n)
    print(1)
    print(n_eigen_vectors.shape)
    # res = projecting(centered_A, n_eigen_vectors)
    return n_eigen_vectors.transpose()






# mglearn.plots.plot_pca_illustration()



# Represent some of our images
# fig, axes = plt.subplots(4,4, sharex=True, sharey=True, figsize=(8, 10))
# faceimages = list(faces.values())[-16:]  # take last 16 images
# for i in range(16):
#     axes[i % 4][i // 4].imshow(faceimages[i], cmap="gray")
# plt.show()

faces_shape = list(faces.values())[0].shape

def eigenfaces_matrix(faces):
    # Let`s observe the shape of our images

    # print(faces_shape)
    # print(faces.items())
    facematrix = []
    facelabel = []
    for key, val in faces.items():
        # if '/s40' in key:
        #     continue
        # if '/s39/10.pgm' in key:
        #     continue
        facematrix.append(val.flatten())
        facelabel.append(key.split('/')[6])

    facematrix = np.array(facematrix)
    return facematrix, facelabel

facematrix, facelabel = eigenfaces_matrix(faces)
training_matrix, _ = eigenfaces_matrix(test_faces)

# print(facematrix)

pca = PCA().fit(facematrix)

train_pca = PCA().fit(training_matrix)


# calculation of number of principal axis
number = 0
s = pca.explained_variance_ratio_[number]
while s < 0.9:
    number += 1
    s+= pca.explained_variance_ratio_[number]
# print(number)

# print(pca.explained_variance_ratio_)
# print(pca.components_[0].shape)

# Take the first K principal components as eigenfaces
n_components = 50
# eigenfaces = pca.components_[:n_components]
train_eigenfaces = train_pca.components_[:n_components]

eigenfaces = PcA(facematrix, n_components)
# print(eigenfaces)
# print(eigenfaces[0])
# print(train_eigenfaces)
# print()
# print(accuracy_score(train_eigenfaces, eigenfaces, normalize=True))
# Show the first 16 eigenfaces
# fig, axes = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(8, 10))
# for i in range(16):
#     axes[i % 4][i // 4].imshow(eigenfaces[i].reshape(faces_shape), cmap="gray")
# plt.show()

weights = []
for i in range(facematrix.shape[0]):
    weight = []
    for j in range(n_components):
        w = eigenfaces[j] @ (facematrix[i] - pca.mean_)
        weight.append(w)
    weights.append(weight)

# print(len(weights[0]))
# print(weights.shape)
# Test on out-of-sample image of existing class
query = test_faces["D:/UCU/Now/Linear Algebra/Face recognition/Testing/s35/9.pgm"].reshape(1,-1)

query_weight = eigenfaces @ (query - pca.mean_).T

# print(query_weight.shape)
# diff = [i - query_weight.T for i in weights]
euclidean_distance = np.linalg.norm(weights - query_weight.T, axis=1)
# print(len(euclidean_distance))
# euclidean_distance = np.linalg.norm(diff, axis=0)
# print(euclidean_distance)
best_match = np.argmin(euclidean_distance)
print("Best match %s with Euclidean distance %f" % (facelabel[best_match], euclidean_distance[best_match]))
# print(facelabel)
# Visualize
fig, axes = plt.subplots(1,2,sharex=True,sharey=True,figsize=(8,6))
axes[0].imshow(query.reshape(faces_shape), cmap="gray")
axes[0].set_title("Request")
axes[1].imshow(facematrix[best_match].reshape(faces_shape), cmap="gray")
axes[1].set_title("Best match")
plt.show()

def test(test_images):
    correct = 0
    total = 0
    for each in test_images:
        total += 1
        query = test_images[each].reshape(1,-1)
        query_weight = eigenfaces @ (query - pca.mean_).T

        # diff = [i - query_weight.T for i in weights]
        # print(diff)
        euclidean_distance = np.linalg.norm(weights - query_weight.T, axis=1)
        # print(len(euclidean_distance))
        # euclidean_distance = np.linalg.norm(diff, axis=0)
        # print(euclidean_distance)
        best_match = np.argmin(euclidean_distance)
        # print("Best match %s with Euclidean distance %f" % (facelabel[best_match], euclidean_distance[best_match]))
        if each.split('/')[-2] == facelabel[best_match]:
            correct += 1
        # else:
        #     fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 6))
        #     axes[0].imshow(query.reshape(faces_shape), cmap="gray")
        #     axes[0].set_title("Query")
        #     axes[1].imshow(facematrix[best_match].reshape(faces_shape), cmap="gray")
        #     axes[1].set_title("Best match")
        #     plt.show()
    return correct/total

print('Accuracy:')
print(test(test_faces))


