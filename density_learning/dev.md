# Density Learning

The goal for this step of the algorithm is to estimate P(J | man(I)) given the samples provided. Note that what exactly 
is needed for steps 2 and 3 are vectors of estimated densities:
1. [P(j_1 | man(i_k)), ..., P(j_N | man(i_k))] for all k in \{1, ..., N\} 
2. [P(j_k | man(i_1)), ..., P(j_k | man(i_N))] for all k in \{1, ..., N\} 

The reference paper used dimensionality reduction with autoencoders followed by kernel density estimation. They also 
mention RNADE (https://arxiv.org/abs/1306.0186) as a more sophisticated approach they have not tried.