#!/usr/bin/env/python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix, dia_matrix, diags
from scipy.sparse.linalg import eigs, eigsh
from timeit import default_timer as timer
from multiprocessing import Pool
from argparse import ArgumentParser
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

parser = ArgumentParser(description="Tool for solving radially-symmetric TDSE under\n external time-dependent potentials")
parser.add_argument('num_x_pts', type=int, help='(int) number of points to use in radial direction.\n(fixed end points of -10,10)')
parser.add_argument('dt', type=float, help='(float) time-step size for time propagation')
parser.add_argument('num_time_steps', type=int, help='(float) number of time steps to propagate through')
parser.add_argument('--num_jobs', '-j', type=int, help='(int) number of jobs to split propagation over', default=3)
parser.add_argument('--num_workers', '-w', type=int, help='(int) number of workers to perform the jobs\n(recommend <= num_jobs)', default=3)
parser.add_argument('--show_improvement', '-i', action='store_true', help='(flag) run non-parallel also and show improvement')
parser.add_argument('--plot', '-p', action='store_true', help='(flag) plot the evolved state \n(will plot both if show_improvement is used)')
args = parser.parse_args()

"""
Setting up global variables:
"""

assert args.dt < 5/args.num_x_pts, "your dt is too big compared to your dx (stability issues); increase num_x_pts or decrease dt"
# Note: above stability check found heuristically

x = np.linspace(-10,10,args.num_x_pts)
v = 200
n = 2
b = 2

@np.vectorize
def potential_softcore(xk, t):
    numerator = -v
    denominator = (np.abs(xk)**n + b**n)**(1/n)
    return numerator/denominator

@np.vectorize
def potential_linear_with_time(xk, t):
    alpha = 5
    numerator = -v
    denominator = (np.abs(xk)**n + b**n)**(1/n)
    return numerator/denominator + alpha*t*xk

@np.vectorize
def potential_oscillating_with_time(xk,t):
    omega = 20
    numerator = -v
    denominator = (np.abs(xk)**n + b**n)**(1/n)
    return numerator/denominator + (0 if t<0.01 else 20*np.sin(t*omega)*xk*np.exp(-0.5*((t)/0.4)**2))

"""
Choose which potential function to use: 
"""
potential = potential_linear_with_time

def gen_diag_Hamiltonian(x_arr):
    
    dx2 = -1/(2*(np.abs(x_arr[0]-x_arr[1])**2))
    
    centre_diag = -(5/2)*np.ones_like(x_arr)*dx2
    one_off_diag = (4/3)*np.ones_like(x_arr[:-1])*dx2    
    two_off_diag = -(1/12)*np.ones_like(x_arr[:-2])*dx2
    
    H = diags([centre_diag,one_off_diag,one_off_diag,two_off_diag,two_off_diag],[0,1,-1,2,-2])
    return H

def gen_diag_V(x_arr, potential_func, t):
    V = potential_func(x_arr, t)
    return diags([V],[0])

def gen_full_Hamiltonian(x_arr, potential_func, t=0): 
    return gen_diag_Hamiltonian(x_arr) + gen_diag_V(x_arr, potential_func, t)

H_baseline = gen_diag_Hamiltonian(x).tocsr()
H = gen_full_Hamiltonian(x, potential, 0)

biggest_abs_val = 100
eig_vals,eig_vecs = eigsh(H, k=5, sigma=-biggest_abs_val)

"""
Parallelised F.D. Propagator;
> Define n_jobs to be number of jobs to run in parallel
> split eigenstates and hamiltonian into 'chunks' - to have n_jobs chunks
"""
n_jobs = args.num_jobs
dt = args.dt

def time_evo_operator(eig_val, t):
    return np.exp(-1j*eig_val*t)

original_eigstate = eig_vecs[:,0]

evolve_o_eigstate_no_ext_n_times = lambda n: [original_eigstate * time_evo_operator(eig_vals[0], dt)**i for i in range(0,n)]

"""
6th order time propogation results:
"""
sixth_order_coeffs = np.array([-49/20, 6, -15/2, 20/3, -15/4, 6/5, -1/6])/dt

def chunk_baseline_Hamiltonian(Hamiltonian, n_jobs):
    """
    Baseline: Hamiltonian with no potential term on central diagonal yet
    """
    global x
    
    chunked_H = []
    x_chunks = []
    
    sizes = [Hamiltonian.shape[0]//n_jobs for job in range(n_jobs)]
    for overshot in range(Hamiltonian.shape[0] % n_jobs):
        sizes[overshot] += 1
        
    i = 0
    for j in sizes:
        chunked_H.append(Hamiltonian[i:i+j,i:i+j])
        x_chunks.append(x[i:i+j])
        i+=j
    
    return chunked_H, x_chunks

chunked_baseline_Hamiltonian, x_chunks = chunk_baseline_Hamiltonian(H_baseline, n_jobs) 

def evolve_state(dt, previous_eigstates, coeffs, t, job_num=None):
    global chunked_baseline_Hamiltonian, x_chunks, x, potential
    
    if job_num is None:
        H = gen_full_Hamiltonian(x, potential, t)
    else:
        H_baseline = chunked_baseline_Hamiltonian[job_num]
        V = gen_diag_V(x_chunks[job_num], potential, t)
        H = H_baseline + V
        
    evolved_part = (H*previous_eigstates[-1])
    new_eigstate = 1j*dt*evolved_part
    for i,c in enumerate(coeffs[1:]):
        new_eigstate -= c*previous_eigstates[-i-1]
    new_eigstate /= coeffs[0]
    return new_eigstate

previous_eigstates = evolve_o_eigstate_no_ext_n_times(6)

full_states = previous_eigstates.copy()
p_e_s = [np.array_split(state, n_jobs) for state in previous_eigstates]
chunked_states = [[i[j] for i in p_e_s] for j in range(n_jobs)] 

def time_step(t, previous_eigstates, job_num=None):
    """
    Works with full eigenstates, or with chunked states
    """
    previous_eigstates.append(evolve_state(dt, previous_eigstates, sixth_order_coeffs, t, job_num))
    return previous_eigstates[1:]

"""
Comparison of speed for parallelised and non-parallelised verisons
"""

def propagate(chunk, num_dts, job_num=None):
    time = 0
    for t in range(num_dts):
        chunk = time_step(time, chunk, job_num)
        time += dt
    return chunk

def task(job_chunk):
    global args
    job, chunk = job_chunk
    return propagate(chunk, args.num_time_steps, job)

pool = Pool(args.num_workers)
par_start = timer()
evolved_chunked_states = pool.map(task, ((job,chunk) for job,chunk in enumerate(chunked_states)))
par_end = timer()

if args.show_improvement:
    sing_start = timer()
    evolved_full_states = propagate(previous_eigstates, args.num_time_steps)
    sing_end = timer()

    print('parallel ({} jobs, {} workers): {:.3f}s, singular: {:.3f}s, improvement: {:.3f}x'.format(args.num_jobs,
                                                                args.num_workers,
                                                                par_end-par_start, 
                                                                sing_end-sing_start, 
                                                                (sing_end-sing_start)/(par_end-par_start)
                                                                ))

    combined = np.concatenate([evolved_chunked_states[i][-1] for i in range(len(evolved_chunked_states))])

    print('mean differences: {}'.format(np.mean(evolved_full_states[-1] - combined).real))

else:
    print('({} jobs, {} workers) time taken: {:.3f}s'.format(args.num_jobs, args.num_workers, par_end-par_start))

if args.plot:
    fig = plt.figure()
    ax = plt.subplot(111)
    if args.show_improvement:
        plt.plot(x, evolved_full_states[-1], 'x', alpha=0.5, label='non-parallel')
    plt.plot(x, combined, 'o', alpha=0.5, label='parallel')
    plt.legend(loc='best')
    plt.show()

