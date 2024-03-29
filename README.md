### Chaos Game Representation for biological sequence analysis
    
Here, I implemented the Chaos Game Representation (CGR) for DNA, RNA and protein sequences in Python. For a given sequence, a user may create, plot or save the appropriate representation. Additionally:
- In the case of RNA, the secondary structure may be included (in Vienna format).
- In the case of protein, a user may use an outer representation (it is possible to specify the 2D features used to describe amino acids; by default, there are hydrophobicity and hydrophilicity). It is also possible to generate the representation using the embedding concept from natural language processing. For this purpose, an example network is presented.

If a user specifies the representation for proteins, it should be provided in the following amino acids order:

ACDEFGHIKLMNPQRSTVWY

as a list of lists.

Moreover, it is possible to generate the phylogenetic tree according to CGR and:
- Discrete Fourier Transform (DFT),
- Structural Similarity Index Measure (SSIM),
- Hurst exponent.

To illustrate how to use these functionalities, there is an Example.ipynb file.
