# Learning Quantum States with Deep Vision Transformers

Neural quantum states (NQS) are a powerful framework for representing and learning quantum states in many-body physics [1]. Recently, transformer architectures, specifically Deep Vision Transformers (Deep ViTs), have shown potential in handling complex, high-dimensional data with locality properties similar to lattice-based quantum systems. Deep ViTs' capacity for spatial feature extraction makes them a promising candidate for learning ground states and characterizing phase transitions in quantum lattice systems [2,3].

This proposal explores the use of Deep ViTs to approximate ground-state wavefunctions of quantum systems on a lattice. Our approach leverages the ability of transformers to handle dependencies across lattice sites, allowing for efficient representations of quantum states in higher dimensions and capturing long-range correlations. We aim to use Deep ViTs as NQS to learn ground states for lattice-based quantum systems, such as the 2D Heisenberg model. We propose a method that employs variational energy minimization to tune the network’s parameters for high-fidelity ground state approximation.


[1] Giuseppe Carleo and Matthias Troyer. “Solving the Quantum Many-Body Problem with Ar-
tificial Neural Networks”. In: Nature 558 (2017), pp. 446–449. doi: 10.1038/nature24270.

[2] Xiaodong Cao, Zhicheng Zhong, and Yi Lu. Vision Transformer Neural Quantum States for
Impurity Models. 2024. arXiv: 2408.13050.

[3] Luciano Loris Viteritti, Riccardo Rende, and Federico Becca. “Transformer Variational Wave
Functions for Frustrated Quantum Spin Systems”. In: Physical Review Letters 130.23 (June
2023). issn: 1079-7114. doi: 10.1103/physrevlett.130.236401.
