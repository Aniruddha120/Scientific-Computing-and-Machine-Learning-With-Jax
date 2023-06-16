import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap, pmap,make_jaxpr,lax,random, device_put ##from lax to device put all are low level API. lax is for XLA
import matplotlib.pyplot as plt
import torch


x_c = torch.arange(1,1000,10)
y_c = 2*np.sin(x_c)*np.cos(x_c)
plt.plot(x_c,y_c)


p_c = np.linspace(1,10,1000)
o_c = 2*np.sin(p_c)*np.cos(p_c)
plt.plot(x_c,y_c,p_c,o_c)

#creating graph

p_c = np.arange(1,10,1000)
o_c = 2*np.sin(p_c)*np.cos(p_c)
plt.plot(x_c,y_c,p_c,o_c)