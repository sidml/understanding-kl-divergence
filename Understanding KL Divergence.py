#!/usr/bin/env python
# coding: utf-8

# I got curious about KL Divergence after reading the Variational Auto Encoder Paper. So, I decided to investigate it to get a better intuition. 
# 
# KL Divergence is a measure of how one probability distribution ($P$) is different from a second probability distribution ($Q$). If two distributions are identical, their KL div. should be 0. Hence, by minimizing KL div., we can find paramters of the second distribution ($Q$) that approximate $P$.
# 
# In this post i try to approximate the distribution $P$ (sum of two gaussians) by minimizing its KL divergence with another gaussian distribution $Q$.
# 

# # Loading Libraries

# In[1]:


import pdb
import numpy as np
import torch
from torch.autograd import grad
import torch.nn.functional as F
import matplotlib.pyplot as plt

# for animation
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.animation
from IPython.display import Image

import warnings
warnings.filterwarnings('ignore')

from typing import Dict, List, Tuple


# # Constructing Gaussians

# Pytorch provides easy way to obtain samples from a particular type of distribution. You can find many types of commonly used distributions in torch.distributions
# Let us first construct two gaussians with $$\mu_{1}=-5,\sigma_{1}=1$$ and $$\mu_{1}=10, \sigma_{1}=1$$

# In[2]:


mu1,sigma1 = -5,1
mu2,sigma2 = 10,1

gaussian1 = torch.distributions.Normal(mu1,sigma1) 
gaussian2 = torch.distributions.Normal(mu2,sigma2)


# # Sanity Check
# Let us sample the distributions at some points to verify if its a gaussian with expected parameters

# In[3]:


plt.figure(figsize=(14,4))
x = torch.linspace(mu1-5*sigma1,mu1+5*sigma1,1000)
plt.subplot(1,2,1)
plt.plot(x.numpy(),gaussian1.log_prob(x).exp().numpy())
plt.title(f'$\mu={mu1},\sigma={sigma1}$')

x = torch.linspace(mu2-5*sigma2,mu2+5*sigma2,1000)
plt.subplot(1,2,2)
plt.plot(x.numpy(),gaussian2.log_prob(x).exp().numpy())
plt.title(f'$\mu={mu2},\sigma={sigma2}$')

plt.suptitle('Plotting the distributions')


# The above figure shows that the distributions have been correctly constructed.
# 
# Let us add the gaussians and generate a new distribution, $P(x)$. 
# 
# Our aim will be to approximate this new distribution
# using another gaussian $Q(x)$. We will try to find the parameters $\mu_{Q},\sigma_{Q}$ by minimizing the KL divergence between the distributions $P(x)$ and $Q(x)$

# In[4]:


plt.figure(figsize=(14,4))
x = torch.linspace(-mu1-mu2-5*sigma1-5*sigma2,mu1+mu2+5*sigma1+5*sigma2,1000)
px = gaussian1.log_prob(x).exp() + gaussian2.log_prob(x).exp()
plt.subplot(1,2,2)
plt.plot(x.numpy(),px.numpy())
plt.title('$P(X)$')


# ## Constructing $Q(X)$
# 
# We will be using a gaussian distribution to approximate $P(X)$. We don't know the optimal paramters that will best approximate the distribution $P(x)$.
# 
# So, let us simply take $\mu=0,\sigma=1$ as our starting point. 
# 
# We could chosen better values, since we already have prior knowledge about the distribution that we are trying to approximate $(P(x))$. But this is mostly not the case in real scenarios

# In[5]:


mu = torch.tensor([0.0])
sigma = torch.tensor([1.0])

plt.figure(figsize=(14,4))
x = torch.linspace(-mu1-mu2-5*sigma1-5*sigma2,mu1+mu2+5*sigma1+5*sigma2,1000)
Q = torch.distributions.Normal(mu,sigma) # this should approximate P, eventually :-)
qx = Q.log_prob(x).exp()
plt.subplot(1,2,2)
plt.plot(x.numpy(),qx.detach().numpy())
plt.title('$Q(X)$')


# ## KL-Divergence
# $$D_{KL}(P(x)||Q(X)) = \sum_{x \in X} P(x) \log(P(x) / Q(x))$$
# 

# ### Computing in pytorch

# Pytorch provides function for computing KL Divergence. You can read more about it [here](https://pytorch.org/docs/stable/nn.html#torch.nn.functional.kl_div).
# 
# The thing to note is that the input given is expected to contain log-probabilities. The targets are given as probabilities (i.e. without taking the logarithm).
# 
# So, the first argument to the function will be Q and second argument will be P (the target distribution).
# 
# Also, we have to be careful about numerical stabilty. 

# In[6]:


px = gaussian1.log_prob(x).exp() + gaussian2.log_prob(x).exp()
qx = Q.log_prob(x).exp()
F.kl_div(qx.log(),px)


# The Divergence is Infinity :-) I think this issue is caused when we exponentiate and then again take log. Using the log values directly seems to be fine.

# In[7]:


px = gaussian1.log_prob(x).exp() + gaussian2.log_prob(x).exp()
qx = Q.log_prob(x)
F.kl_div(qx,px)


# In[8]:


def optimize_loss(px: torch.tensor, loss_fn: str, muq: float = 0.0, sigmaq: float = 1.0,                  subsample_factor:int = 3,mode:str = 'min') -> Tuple[float,float,List[int]]:
    
    mu = torch.tensor([muq],requires_grad=True)
    sigma = torch.tensor([sigmaq],requires_grad=True)    

    opt = torch.optim.Adam([mu,sigma])

    loss_val = []
    epochs = 10000

    #required for animation
    all_qx,all_mu = [],[]
    subsample_factor = 3 #have to subsample to reduce memory usage

    torch_loss_fn = getattr(F,loss_fn)
    for i in range(epochs):
        Q = torch.distributions.Normal(mu,sigma) # this should approximate P
        if loss_fn!='kl_div': # we need to exponentiate q(x) for these and few other cases
            qx = Q.log_prob(x).exp()
            all_qx.append(qx.detach().numpy()[::subsample_factor])
        else:
            qx = Q.log_prob(x)
            all_qx.append(qx.exp().detach().numpy()[::subsample_factor])
            
        if mode=='min':
            loss = torch_loss_fn(qx,px)
        else:
            loss = -torch_loss_fn(qx,px,dim=0)
    #   backward pass
        opt.zero_grad()
        loss.backward()    
        opt.step()
        loss_val.append(loss.detach().numpy())
        all_mu.append(mu.data.numpy()[0])
        
        
        if i%(epochs//10)==0:
            print('Epoch:',i,'Loss:',loss.data.numpy(),'mu',mu.data.numpy()[0],'sigma',sigma.data.numpy()[0])


    print('Epoch:',i,'Loss:',loss.data.numpy(),'mu',mu.data.numpy()[0],'sigma',sigma.data.numpy()[0])
    
    plt.figure(figsize=(14,6))
    plt.subplot(2,2,1)
    plt.plot(loss_val)
    plt.xlabel('epoch')
    plt.ylabel(f'{loss_fn} (Loss)')
    plt.title(f'{loss_fn} vs epoch')
    
    plt.subplot(2,2,2)
    plt.plot(all_mu)
    plt.xlabel('epoch')
    plt.ylabel('$\mu$')
    plt.title('$\mu$ vs epoch')
    
    return mu.data.numpy()[0],sigma.data.numpy()[0],all_qx


# In[9]:


x = torch.linspace(-mu1-mu2-5*sigma1-5*sigma2,mu1+mu2+5*sigma1+5*sigma2,1000)
px = gaussian1.log_prob(x).exp() + gaussian2.log_prob(x).exp()
mu,sigma,all_qx = optimize_loss(px, loss_fn='kl_div', muq = 0.0, sigmaq = 1.0)


# In[10]:


def create_animation(x:torch.tensor,px:torch.tensor,all_qx:List,subsample_factor:int = 3,                     fn:str = 'anim_distr.gif') -> None:

    # create a figure, axis and plot element 
    fig = plt.figure() 
    ax = plt.axes(xlim=(x.min(),x.max()), ylim=(0,0.5)) 
    text = ax.text(3,0.3,0)
    line1, = ax.plot([], [], color = "r")
    line2, = ax.plot([], [], color = "g",alpha=0.7)

    def animate(i):    
    #     non uniform sampling, interesting stuff happens fast initially
        if i<75:
            line1.set_data(x[::subsample_factor].numpy(),all_qx[i*50])
            text.set_text(f'epoch={i*50}')
            line2.set_data(x[::subsample_factor].numpy(),px.numpy()[::subsample_factor])
        else:
            line1.set_data(x[::subsample_factor].numpy(),all_qx[i*100])
            text.set_text(f'epoch={i*100}')
            line2.set_data(x[::subsample_factor].numpy(),px.numpy()[::subsample_factor])

        return [line1,line2]

    ani = matplotlib.animation.FuncAnimation(fig,animate,frames=100 
                                   ,interval=200, blit=True)

    fig.suptitle(f'Minimizing the {fn[:-3]}')
    ax.legend(['Approximation','Actual Distribution'])
    # save the animation as gif
    ani.save(fn, writer='imagemagick', fps=10) 


# In[11]:


# %% capture if you dont want to display the final image
ani = create_animation(x,px,all_qx,fn='kl_div.gif')
Image("../working/kl_div.gif")


# Let us try to see what we get when we try to solve the Mean Squared Distance between $P$ and $Q$

# In[12]:


x = torch.linspace(-mu1-mu2-5*sigma1-5*sigma2,mu1+mu2+5*sigma1+5*sigma2,1000)
px = gaussian1.log_prob(x).exp() + gaussian2.log_prob(x).exp()
mu,sigma,all_qx = optimize_loss(px, loss_fn='mse_loss', muq = 0.0, sigmaq = 1.0)


# In[13]:


fn = 'mse_loss_mean0.gif'
ani = create_animation(x,px,all_qx,fn=fn)
Image(f"../working/{fn}")


# We can see that the result is very different from the KL divergence case. We are converging towards one of the gaussians, no middle ground !
# 
# Also, you can try expeimenting with different initial values for $\mu_{Q}$. If you choose a value that is closer to 10 (mean of the second gaussian), it will converge towards it.

# In[15]:


x = torch.linspace(-mu1-mu2-5*sigma1-5*sigma2,mu1+mu2+5*sigma1+5*sigma2,1000)
px = gaussian1.log_prob(x).exp() + gaussian2.log_prob(x).exp()
mu,sigma,all_qx = optimize_loss(px, loss_fn='mse_loss', muq = 5.0, sigmaq = 1.0)

fn = 'mse_loss_mean5.gif'
ani = create_animation(x,px,all_qx,fn=fn)
Image(f"../working/{fn}")


# You can easily very this to be the case for L1 loss also. 
# 
# Now, let us try to see what we get when we try to maximize the cosine similarity between two distributions.
# 

# In[ ]:


x = torch.linspace(-mu1-mu2-5*sigma1-5*sigma2,mu1+mu2+5*sigma1+5*sigma2,1000)
px = gaussian1.log_prob(x).exp() + gaussian2.log_prob(x).exp()
mu,sigma,all_qx = optimize_loss(px, loss_fn='cosine_similarity', muq = 5.0, sigmaq = 1.0,mode='max')

fn = 'cosine_similarity.gif'
ani = create_animation(x,px,all_qx,fn=fn)
Image(f"../working/{fn}")


# ## Conclusion
# As shown above in 1D case, we converge towards the nearest mean values. In a high dimensional space with multiple valleys, minimizing the MSE/L1 Loss can lead to different results.  In deep learning, we are randomly initializing the weights of neural network. So, it makes sense that we converge towards different local minimas in different runs of the same neural network. 
# Techniques like stochastic weight averaging perhaps improve generalizibility because they offer weights to different local minimas. Its possible that different local minimas encode important information about the dataset.
# 
# In the next post, I will try to explore Wasserstein distance.

# In[ ]:




