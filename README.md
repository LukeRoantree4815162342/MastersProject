# Masters Project: Computational Methods For Ultrafast Quantum Physics

---

## My project focuses on efficiently modelling laser pulse interactions with atoms

To do this;

> I solve (find eigenstates of) the Time-Independent Schrodinger Equation for a given potential distribution (so far using a soft-core potential to numerically simulate a Coulomb potential while avoiding the problem at the centre)

> I pick a state, and temporally propagate it to see how it evolves in time under a time-dependent potential. To simulate a `laser pulse', I add a time-dependent gaussian packet term to the initial soft-core potential.

---

## Optimisations So Far:

### Sparse Storage for Hamiltonian; I use sparse diagonal storage, allowing much larger numbers of spatial points to be used

### Sparse Eigensolver; I use a Krylov subspace based eigensolver that allows sparse inputs, speeding up finding states

### ML For Shift-Inverse Method; The eigensolver has a parameter to allow it to look for eigenstates with eigenvalues near a value provided; I trained a Neural Network to predict the eigenvalue of the Ground State based on the potential distribution, and pass this result into the eigensolver - this speeds it up dramatically

### Finite Difference Propagator; Implemented an arbitrary-order finite difference method time propagator

### Krylov-Subspace Propagator; Implemented an arbitrary-order Arnoldi-based time propagator

### Parallelised The FD Propagator; I used multiprocessing to more efficiently propagate the state through time (speed-ups depending on number of cores available in the computer / server cluster used)

---

## TODO:

### Re-Write In Fortran; In progress, switching over to MPI & OpenMP for parallelisation

### Apply some results found to existing RMT code to investigate if it can similarly be improved, and if it can maintain stability under these
 
