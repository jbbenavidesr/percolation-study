# Percolation Simulation

This project is the starting point for my investigation in site percolation.
I need to more deeply understand how it works and how to study it and be able to
calculate critical exponents.

My work will be to study the results of jigsaw puzzle sequences of being filled,
calculate the critical exponents of Jigsaw puzzles in order to compare it with
different percolation models.

Here I'll start with a script that generates site percolation square lattices.
With them I'll build a script with the Hosen-Kopelman algorithm that takes
a lattice (any) and returns the information of all the clusters in the lattice.

Finally I'll build a module that uses the cluster information and calculates the
critical exponents.

With all thies pieces I'll write a jupyter notebook introducing all site percolation
and studying how it works in order to verify that the exponents and critical points
match the theory and therefore my modules work. With this I can test with some other
models like bond and invasion percolation and finally, use them to measure criticality
of the puzzle results.
