import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.stats as st


# Plot bivariate distribution
def grid2d_pdf(pdf_fun,limits=[-5,5]):
    """Helper function to generate density surface."""
    nb_of_x = 50 # grid size
    x1s = np.linspace(limits[0], limits[1], num=nb_of_x)
    x2s = np.linspace(limits[0], limits[1], num=nb_of_x)
    x1, x2 = np.meshgrid(x1s, x2s) # Generate grid
    x_12 = np.stack((x1,x2),axis=-1)    
    
    # Fill the cost matrix for each combination of weights
    pdf_grid = pdf_fun(x_12) 
            
    return x1, x2, pdf_grid 


def plot_2d_GMM(mean_list,cov_list,alphas,samples=None,limits=[-5,5]):

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
    #fig, ax = plt.subplots()
    if samples is not None:
        ax.scatter(samples[:,0],samples[:,1],color="C3",alpha=0.2)
        
        
    for i in range(0,len(mean_list)):
        plot_2d_normal(mean=mean_list[i],cov=cov_list[i],ax=ax,coeff=alphas[i],limits=limits)


    ax.set_xlabel('$x_1$', fontsize=13)
    ax.set_ylabel('$x_2$', fontsize=13)
    ax.axis([limits[0], limits[1], limits[0], limits[1]])
    ax.set_aspect('equal')
    

def plot_1d_GMM(mean_list=None,var_list=None,alphas=None,samples=None,limits=[-5,5]):

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
    #fig, ax = plt.subplots()
    if samples is not None:
        _ = ax.hist(samples,bins=len(samples)//20,density=True,color="C0",alpha=0.2)
        
    nb_of_x = 500 # grid size
    x1s = np.linspace(limits[0], limits[1], num=nb_of_x)
    if mean_list is not None:
        for i in range(0,len(mean_list)):
            ax.plot(x1s,alphas[i]*st.norm.pdf(x1s,loc=mean_list[i],scale=np.sqrt(var_list[i])),linewidth=2.)

    ax.set_xlabel('$x$', fontsize=13)
    ax.set_ylabel('$p(x)$', fontsize=13)
    #ax.axis([limits[0], limits[1], limits[0], limits[1]])
    #ax.set_aspect('equal')

    
def plot_logp_alpha(log_p,alpha):
    
    fig, [ax1,ax2] = plt.subplots(nrows=1, ncols=2, figsize=(10,5))

    ax1.plot(np.array(range(0,len(log_p))),log_p)
    ax1.set_xlabel('$iteration$', fontsize=13)
    ax1.set_ylabel('$log(p(x))$', fontsize=13)
    ax1.grid()
    
    ax2.bar(np.array(range(0,len(alpha))),height=alpha)
    ax2.set_xlabel('$k$', fontsize=13)
    ax2.set_ylabel('$alpha_k$', fontsize=13)

def plot_2d_normal(mean,cov,ax,coeff=None,limits=[-5,5]):


    pdf_fun = lambda var: st.multivariate_normal.pdf(mean=mean,cov=cov,x=var)
    
    x1, x2, p2 = grid2d_pdf(pdf_fun,limits)
    if coeff is not None:
        p2 = p2 * coeff
    # Plot bivariate distribution
    con = ax.contour(x1, x2, p2, 6, cmap=cm.viridis)
