# Spam Message Filtering : A Comparative Study
## Using Word Embedding and Weighted Aggregation of Words
In this work, we aim to build a comparative analysis to classify messages into spam and non spam labels using machine learning and artificial neuron network models. 

To convert a word into a numeric vector here [Word2Vec](https://en.wikipedia.org/wiki/Word2vec) technique was applied. Under the umbrella of [Word2Vec](https://en.wikipedia.org/wiki/Word2vec) there are 2 different methods defined. Skip-gram and CBOW methods. Both the methods are implemented here. 

Here to get the vector representation of sentence through the constituting words the converted word vectors are aggregated. Usual conventions suggest to have a flat average. In this project we have devised an algorithm which initializes a weight to every word vector. The weight initialization is based on the probability of occurrence of a word in both the classes e.g. spam and non-spam. Such that the algorithm calculates the occurrence of one word in each classes such that the probabilities are $p_1$ and $p_2$. Now if $p_1>p_2$ then the weight will be $\frac{p_1}{p_2+\epsilon}-1$ otherwise the weight will be $\frac{p_2}{p_1+\epsilon}-1$. To avoid denominator to be 0 an arbitrary small number $\epsilon>0$ is taken into consideration.

In the study, we also focused on implementing a famous neural network model i.e., [Radial Basis Neural Network](https://en.wikipedia.org/wiki/Radial_basis_function_network). This is also used in classification problems. But in this case or rather in Natural Language Processing this algorithm is not recommended. 

Overall here the comparison of weighted aggregation and performance of the machine learning models and neural networks have been established.
