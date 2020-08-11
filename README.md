***This code modified from the code avaliable at: "https://github.com/IcarPA-TBlab/MetagenomicDC/blob/master/models/CNN.py"  

CNN_RAI

The successful results of Deep learning algorithms have been reported recently for genome taxonomic classifications of genomic fragments problem. The current convention main approach in the current literature is feeding the oligonucleotide count vectors to Convolutional Neural Networks (CNN) feature
extractors and cascading classifiers on top. From these perspectives, we employed Relative Abundance Index (RAI) representation as DNA fragment representations to be used in CNN classifiers
(CNN-RAI).

Dependences:

Python (2.7.x)
Theano (0.8.2)
Keras library (2.x)

Dataset:

Oxford Nanopore MinION sequencing data belonging to Bacteroides, Klebsiella, Yersinia, Mycobacterium, Clostridium and Escherichia genera were obtained from NCBI read archives and NCBI
sra accession numbers respectively are ERR1898312 ,ERR1474981 SRR5117441, SRR5277601, SRR5344355. 
