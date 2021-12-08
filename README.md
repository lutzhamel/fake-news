# kdnuggets-fake-news
This is a further development of the kdnuggets article on fake news classification by George McIntyre:

https://www.kdnuggets.com/2017/04/machine-learning-fake-news-accuracy.html

In his article McIntyre approaches document classification from a very classical perspective: applying a vector-model to the corpus and then using a Naive Bayes classifier.  Here we take it into the deep learning realm: we apply a deep convolutional network with a traininable word-embedding layer.

We compare the performances of both approaches.  The notebook [fake_news_classification.ipynb](fake_news_classification.ipynb) contains our computational results. The markdown document [report.md](report.md) reports our results in a
more accessible manner.  The results were also posted at [opendatascience](https://opendatascience.com/deep-learning-finds-fake-news-with-97-accuracy).

