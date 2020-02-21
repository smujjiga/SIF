# Sentence embedding using Smooth Inverse Frequency weighting scheme

This is the implementation of SIF prposed in "A Simple but Tough-to-Beat Baseline for Sentence Embeddings" https://openreview.net/forum?id=SyK00v5xx.

This code heavily reuses the implementation available at https://github.com/PrincetonML/SIF

The changes I have done are to make it fast and scalabeld and fix few issues. The code focuses on generating the embeddings using  SIF which can be used in the downstream tasks. 

Check the `test.py` file to understand how to generate the embeddings. 

Download and copy the glove file `glove.840B.300d.txt` to the working direcotyr and run `python test.py`
