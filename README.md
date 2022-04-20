# A selected movielens dataset for The Neural Graph Collaborative Filtering model to test.

In this project, we select a threshold --n to train NDCG model.
A parameter n is a threshold that ranges from 3 to 5.
Only when users' rating > threshold, we consider the movie should be recommended to user.

The Neural Graph Collaborative Filtering (https://doi.org/10.1145/3331184.3331267) is implement by the Pytorch Framework provided by 
(https://medium.com/@yusufnoor_88274/implementing-neural-graph-collaborative-filtering-in-pytorch-4d021dff25f3)