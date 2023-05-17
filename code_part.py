import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
# print(pca)

# calculation of number of principal axis
number = 0
s = pca.explained_variance_ratio_[number]
while s < 0.8:
    number += 1
    s+= pca.explained_variance_ratio_[number]
print(number)

# print(pca.explained_variance_ratio_)
# print(pca.components_[0].shape)

# Take the first K principal components as eigenfaces
n_components = 50
eigenfaces = pca.components_[:n_components]
train_eigenfaces = train_pca.components_[:n_components]
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
query = faces["D:/UCU/Now/Linear Algebra/Face recognition/Training/s40/1.pgm"].reshape(1,-1)

query_weight = eigenfaces @ (query - pca.mean_).T

print(query_weight.shape)
diff = [i - query_weight.T for i in weights]
euclidean_distance = np.linalg.norm(weights - query_weight.T, axis=1)
print(len(euclidean_distance))
# euclidean_distance = np.linalg.norm(diff, axis=0)
# print(euclidean_distance)
best_match = np.argmin(euclidean_distance)
print("Best match %s with Euclidean distance %f" % (facelabel[best_match], euclidean_distance[best_match]))
# Visualize
fig, axes = plt.subplots(1,2,sharex=True,sharey=True,figsize=(8,6))
axes[0].imshow(query.reshape(faces_shape), cmap="gray")
axes[0].set_title("Query")
axes[1].imshow(facematrix[best_match].reshape(faces_shape), cmap="gray")
axes[1].set_title("Best match")
plt.show()



