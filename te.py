#! /usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

# generate the Hamiltonian of the Rabi model
def H_rabi(N,wc,wa,g):
    H_rabi = np.zeros(shape = (2*N,2*N))
    for i in range(N):
        H_rabi[2*(N-i)-1,2*(N-i)-1] = i*wc-wa/2.0
	H_rabi[2*(N-i-1),2*(N-i-1)] = i*wc+wa/2.0
	if i < N-1:
	   H_rabi[2*(N-i)-1,2*(N-i)-4] = g*np.sqrt(i+1)
	   H_rabi[2*(N-i)-4,2*(N-i)-1] = g*np.sqrt(i+1)
	   H_rabi[2*(N-i-1),2*(N-i)-3] = g*np.sqrt(i+1)
	   H_rabi[2*(N-i)-3,2*(N-i-1)] = g*np.sqrt(i+1)
    return H_rabi

# construct the input vector
def psi_in(N):
    psi_in = np.zeros(shape = (2*N,1))
    psi_in[2*(N-1),0]=1
    return psi_in

N = 50
wc = 1
wa = 1
psi = psi_in(N)
dt = 0.005
g0 = 0.5
omega = np.pi/10
t_in = 0
t_out = 100
S = 20001
t = np.linspace(t_in,t_out,S)
P = np.zeros(shape = (S+1,3))
P[0,:] = [t_in,0,1]
H0 = H_rabi(N,wc,wa,0)
K1 = -1j*np.dot(H0,psi)
for i in range(S):
    th = (i+0.5)*dt+t_in
    g1 = g0*np.sin(omega*th)
    H1 = H_rabi(N,wc,wa,g1)
    K2 = -1j*np.dot(H1,psi+K1*dt/2.0)
    K3 = -1j*np.dot(H1,psi+K2*dt/2.0)
    g2 = g0*np.sin(omega*(th+dt/2.0))
    H2 = H_rabi(N,wc,wa,g2)
    K4 = -1j*np.dot(H2,psi+K3*dt)
    psi = psi+(K1+2*K2+2*K3+K4)*dt/6.0
    P[i+1,0] = th+dt/2.0
    psi_e = psi[0:2*N:2]
    psi_g = psi[1:2*N:2]
    P[i+1,1] = sum(abs(psi_e)**2)[0]
    P[i+1,2] = sum(abs(psi_g)**2)[0]
    K1 = -1j*H2*psi
    
x = P[:,0]
y1 = P[:,1]
y2 = P[:,2]
plt.plot(x,y1,'.r',x,y2,'.b')
plt.xlabel('t')
plt.ylabel('possibility')
plt.show()
