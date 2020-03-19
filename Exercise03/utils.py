import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.stats as st
import cv2

def lineardemo():
    # input values
    X = np.arange(10)
    # target results
    Y = 0.5 + 0.3 * X + np.random.randn(X.shape[0]) * 0.2
    # % show raw values
    plt.figure(figsize=(10, 8))
    plt.scatter(X,Y, label='real points')
    # extend with ones
    X_ex = np.concatenate((X.reshape(-1, 1), np.ones((X.shape[0], 1))), axis=1)
    # determine optimal parameters
    theta = np.linalg.lstsq(X_ex, Y)[0]

    X2 = np.array((-1, 11))
    # for testing, again extend
    X2_ex = np.concatenate((X2.reshape(-1, 1), np.ones((X2.shape[0], 1))), axis=1)
    # determine y values
    Y2 = np.dot(X2_ex, theta)

    plt.plot(X2, Y2, label='fitted line')
    plt.legend(loc='best')

def plot_linear_model_and_data(m=None,q=None,data=None):
    
    fig, ax = plt.subplots(nrows=1, ncols=1)
    
    if m is not None and q is not None:
        x_grid = np.linspace(-5,5,10)
        ax.plot(x_grid,x_grid*m + q,color="C1")
    
    if data is not None:
        ax.scatter(data[0],data[1],alpha=0.5,color="C0")
        
    ax.set_xlabel('$x$', fontsize=13)
    ax.set_ylabel('$y$', fontsize=13)
        
    plt.show()
    
def load_and_show_images():
    
    img1 = cv2.imread("materials/tower_2.jpg")
    img2 = cv2.imread("materials/tower_1.jpg")

    plt.imshow(img1[:,:,[2,1,0]])
    plt.show()
    
    plt.imshow(img2[:,:,[2,1,0]])
    plt.show()
    
    return img1,img2

def load_correspondences():
    
    data = np.load("materials/points_img_RANSAC.npz")
    kp1 = data["kp1"]
    kp2 = data["kp2"]
    
    img = data["img"]
    
    fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(24,48))
    ax.imshow(img[:,:,[2,1,0]])
    plt.show()
    
    return kp1,kp2

def estimate_transformation_matrix(kp1,kp2):
    
    H,_ = cv2.findHomography(kp2, kp1,0)
    
    return H



def transform_points(kp1,H):
    
    kp1_hom = np.concatenate((kp1,0*kp1[:,0:1] +1),axis=1)
    
    kp2_hom = H.dot(kp1_hom.transpose()).transpose()
    
    kp2_hom = kp2_hom / kp2_hom[:,2:3]
    kp2 = kp2_hom[:,0:2]
    
    return kp2

def blending(img1,img2,H):
    
    height_merged = img1.shape[0] + 2* img2.shape[0]
    width_merged = img1.shape[1] + 2* img2.shape[1]

    full = np.zeros((height_merged, width_merged, 3))
    full[ img2.shape[0]:-img2.shape[0], img2.shape[1]:-img2.shape[1], :] = img1.copy()
    img1 = full.astype(np.uint8)
        
    panorama1 = img1
    mask1 = (np.sum(panorama1,axis=2) > 0).astype(float)

        
    panorama2 = cv2.warpPerspective(img2, H, (width_merged, height_merged))
    mask2 = (np.sum(panorama2,axis=2) > 0).astype(float)
    #plt.imshow(mask2)
    #plt.imshow(panorama2.astype(int))
    #plt.show()
        
    mask = (mask1 + mask2)
    mask[mask == 2] = 0.5
    #plt.imshow(mask)
    #plt.show()
        
    mask = np.expand_dims(mask,2)
    result= mask*panorama1+mask*panorama2
    rows, cols = np.where(result[:, :, 0] != 0)
    min_row, max_row = min(rows), max(rows) + 1
    min_col, max_col = min(cols), max(cols) + 1
    final_result = result[min_row:max_row, min_col:max_col, :]
    
    return final_result.astype(int),mask[:,:,0]