# My Machine Learning Reading List
Following is my reading list on a variety of topics of modern machine learning. It is a personalized list and reflects my flavor as a statistician, but I hope you can benefit from it.

### Comprehensive Books

* [Elements of Statistical Learning, 2nd](http://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12.pdf) **(ESL, bible book of machine learning among statisticians)** - Book
* [An Introduction To Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/) **(A light version of ESL)** - Book

### Probability & Statistics

* [Statistical Inference, 2nd](https://www.amazon.com/Statistical-Inference-George-Casella/dp/0534243126) **(Excellent introductory book to probability and statistics, my graduate textbook)** - Book
* [Computer Age Statistical Inference](https://web.stanford.edu/~hastie/CASI_files/PDF/casi.pdf) **(Textbook of modern statistical inference written by Efron and Hastie)** - Book
* [Bayesian Data Analysis, 3rd](https://www.amazon.com/Bayesian-Analysis-Chapman-Statistical-Science/dp/1439840954) **(Standard reference to Bayesian statistics)** - Book
* [ASA Statement on Statistical Significance and P-Values](http://amstat.tandfonline.com/doi/pdf/10.1080/00031305.2016.1154108?needAccess=true) **(Surprisingly, many people outside (and even inside) statistics community frequently misinterpret p-values, and this is "official" guide to p-values)** - Article

### Linear Models

* [Linear Model with R](http://www.utstat.toronto.edu/~brunner/books/LinearModelsWithR.pdf) **(Introductory book on linear regression with ample R examples)** - Book
* [Linear Statistical Models, 2nd](https://www.amazon.com/Linear-Statistical-Models-James-Stapleton/dp/0470231467) **(For those who want more math, this is our graduate textbook)** - Book

Some milestone papers on regularized regression
* [A Selective Overview of Variable Selection in High Dimensional Feature Space](https://arxiv.org/pdf/0910.1122.pdf) **(Review paper of variable selection in high dimensional regression written by Jianqing Fan)** - Article
* [Regression Shrinkage and Selection via the Lasso](https://statweb.stanford.edu/~tibs/lasso/lasso.pdf) **(Original paper on Lasso)** - Article
* [Regularization and Variable Selection via the Elastic Net](https://web.stanford.edu/~hastie/Papers/B67.2%20(2005)%20301-320%20Zou%20&%20Hastie.pdf) **(Original paper on elastic net)** - Article
* [Model Selection and Estimation in Regression with Grouped Variables](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.366.4278&rep=rep1&type=pdf) **(Variable selection can be performed at group level as well)** - Article
* [Exact Post-Selection Inference, with Application to the Lasso](https://arxiv.org/abs/1311.6238.pdf) **(Statistical inference on Lasso)** - Article

### Bagging & Boosting

* [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/pdf/1603.02754.pdf) **(A "scalable, portable and distributed gradient boosting library", winning algorithms for many Kaggle competitions)** - Article
* [Welcome to LightGBM’s documentation!](https://lightgbm.readthedocs.io/en/latest/) **(Another industry-level gradient boosting library written by Microsoft, 10 times faster than xgboost)** - Website
* [Fighting Biases with Dynamic Boosting](https://arxiv.org/pdf/1706.09516.pdf) **(Paper for Catboost, a recent gradient boosting library that outperforms in accuracy, but has slower training than competing kits such as xgboost)** - Article
* [Deep Forest: Towards An Alternative to Deep Neural Networks](https://arxiv.org/pdf/1702.08835.pdf) **(I wouldn't say an alternative to deep learning, but definitely very interesting idea furthering tree-based methods)** - Article

### Stacking

Rather than bagging and boosting that combines "weak learners", the modern approach is to create an ensemble of a well-chosen collection of strong yet diverse models.
* [Stacked Regressions](http://statistics.berkeley.edu/sites/default/files/tech-reports/367.pdf) **(One of the original paper on stacking models written by Breiman)** - Article
* [Super Learner in Prediction](http://biostats.bepress.com/cgi/viewcontent.cgi?article=1226&context=ucbbiostat) **(Extension to general loss functions)** - Article

### Clustering

* [Data Clustering: A Review](https://www.cs.rutgers.edu/~mlittman/courses/lightai03/jain99data.pdf) **(Review of classical clustering methods)** - Article
Two clustering algorithms particularly interests me
* [On Spectral Clustering: Analysis and an Algorithm](http://ai.stanford.edu/~ang/papers/nips01-spectral.pdf) **(Spectral clustering)** - Article
* [Clustering by Passing Messages Between Data Points](http://www.psi.toronto.edu/affinitypropagation/FreyDueckScience07.pdf) **(Affinity propagation clustering)** - Article

### Deep Learning

#### Tutorial & Overview
* [A Tutorial on Deep Learning Part 1: Nonlinear Classifiers and The Backpropagation Algorithm](http://ai.stanford.edu/~quocle/tutorial1.pdf) - Article
* [A Tutorial on Deep Learning Part 2: Autoencoders, Convolutional Neural Networks and Recurrent Neural Networks](http://ai.stanford.edu/~quocle/tutorial2.pdf) - Article
* [Deep Learning Specialization: Master Deep Learning, and Break into AI](https://www.coursera.org/specializations/deep-learning) **(Andrew Ng's deep learning course on Coursera, the exercises are excellent)** - Website
* [Deep Learning - An MIT Press book](http://www.deeplearningbook.org/) **(Comprehensive book by the big three)** - Book
* [Deep Learning](https://www.nature.com/articles/nature14539) **(Review paper on Nature also by the big three)** - Article

#### Models & Architectures
I don't follow closely on the lastest trend of DL, and following are simply what interest me
* [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shit](https://arxiv.org/pdf/1502.03167v3.pdf) **(Batch normalization)**- Article
* [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf) **(Dropout)** - Article
* [An Overview of Gradient Descent Optimization Algorithms](https://arxiv.org/pdf/1609.04747.pdf) **[Nice review on gradient descent variations]** - Article
* [Searching for Activation Functions](https://arxiv.org/pdf/1710.05941.pdf) **(A recently proposed activation function that outperforms ReLU)** - Article
* [ImageNet Classification with Deep Convolutional Neural Networks](https://www.nvidia.cn/content/tesla/pdf/machine-learning/imagenet-classification-with-deep-convolutional-nn.pdf) **(CNN, one of the milestones in DL)** - Article
* [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) **(For an introduction to LSTM, I recommend this very clearly-written blog)** - Website
* [Generative Adversarial Nets](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf) **(GAN, fairly cool idea)** - Article

#### Platform
* [Keras Documentation](https://keras.io/) **(My favorite high-level DL API)** - Website

### Reinforcement Learning

* [Reinforcement Learning: An Introduction, 2nd](http://ufal.mff.cuni.cz/~straka/courses/npfl114/2016/sutton-bookdraft2016sep.pdf) **(Standard textbook on reinforcement learning)**- Book
* [UCL Course on Reinforcement Learning](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html) **(The lecture David Silver leads DeepMind and is lead researcher on AlphaGo)** - Website
* [CS 294: Deep Reinforcement Learning](http://rll.berkeley.edu/deeprlcourse/) - Website
* [Mastering the Game of Go with Deep Neural Networks and Tree Search](https://gogameguru.com/i/2016/03/deepmind-mastering-go.pdf) **(AlphaGo prototype)** - Article

### Natural Language Processing

### Python & R

I intentionally didn't include any advanced reference, believing that the best way to learn programming is to practice once you get yourself started. There are tons of Q&A online (like at Stack Exchange) which you shall refer to.
* [Python for Data Analysis, 2nd](http://shop.oreilly.com/product/0636920050896.do) **(Written by the author of numpy)** - Book
* [An Introduction to R](https://cran.r-project.org/doc/manuals/r-release/R-intro.pdf) **(Official R documentation and best tutorial for beginners to get started)** - Article
* [The caret Package](http://topepo.github.io/caret/index.html) **(One of my favorite R packages that streamlines the process for creating predictive models)** - Website

### Big Data

* [Challenges of Big Data Analysis](https://arxiv.org/pdf/1308.1479.pdf) **(Review paper written by Jianqing Fan)** - Article
* [A Split-and-Conquer Approach for Analysis of Extraordinarily Large Data](http://www3.stat.sinica.edu.tw/sstest/oldpdf/A24n49.pdf) **(Parallelizing regularized GLM, written by my research group at Rutgers)** - Article
* [A Communication-Efficient Parallel Algorithm for Decision Tree](https://arxiv.org/pdf/1611.01276.pdf) **(Parallelizing tree building)** - Article
* [MapReduce: Simplified Data Processing on Large Clusters](https://static.googleusercontent.com/media/research.google.com/en//archive/mapreduce-osdi04.pdf) **(Original research paper of MapReduce by Jeff Dean)** - Article
* [Spark: Cluster Computing with Working Sets](https://www.usenix.org/legacy/event/hotcloud10/tech/full_papers/Zaharia.pdf) **(Original research paper introducing the concept of Spark)** - Article
* [Resilient Distributed Datasets: A Fault-Tolerant Abstraction for In-Memory Cluster Computing](https://www.usenix.org/system/files/conference/nsdi12/nsdi12-final138.pdf) **(Original research paper introducing the concept of RDD)** - Article

### Miscellaneous
* [A Few Useful Things to Know about Machine Learning](https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf) - Article
* [Model Order Selection: A Review of Information Criterion Rules](http://www.sal.ufl.edu/eel6935/2008/01311138_ModelOrderSelection_Stoica.pdf) **(Review of various information criterions)** - Article
* [Do we Need Hundreds of Classifiers to Solve Real World Classification Problems?](http://jmlr.org/papers/volume15/delgado14a/delgado14a.pdf) **(Interesting paper comparing 179 classification models where random forest wins, although I would put a question mark on the conclusion)** - Article
