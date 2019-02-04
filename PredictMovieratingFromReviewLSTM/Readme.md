# Predict the rating of movies from users' reveiw using RNN LSTM model

Here, we have a large Movie Review Dataset (often referred to as the IMDB dataset) that is provided by by Stanford researchers and was used in a 2011 paper. 
http://ai.stanford.edu/~amaas/data/sentiment/

The data set contains 25,000 highly-polar movie reviews for training and the same amount for testing. Each movie review is a variable sequence of words and and we are building a model to classify the sentiment of each movie review. In other words, the problem is to determine whether a given movie review has a positive or negative sentiment.

We have some sequence of inputs over space and we want to predict a category for the sequence,
 which is here a positive or negative sentiment.
The solution here is to use a Sequence Classification with LSTM Recurrent Neural Networks
 in Python with Keras. Keras provides access to the IMDB dataset built-in. The imdb.load_data() function allows you to load the dataset in a format that is ready for use in neural network and deep learning models. 

The words have been replaced by integers that indicate the ordered frequency of each word in the dataset. The sentences in each review are therefore comprised of a sequence of integers.

### Word Embedding

We will map each movie review into a real vector domain, a popular technique when working with text called word embedding. This is a technique where words are encoded as real-valued vectors in a high dimensional space, where the similarity between words in terms of meaning translates to closeness in the vector space.

Keras provides a convenient way to convert positive integer representations of words into a word embedding by an Embedding layer.

We will map each word onto a 32 length real valued vector. We will also limit the total number of words that we are interested in modeling to the 5000 most frequent words, and zero out the rest. Finally, the sequence length (number of words) in each review varies, so we will constrain each review to be 500 words, truncating long reviews and pad the shorter reviews with zero values.

Now that we have defined our problem and how the data will be prepared and modeled, we are ready to develop an LSTM model to classify the sentiment of movie reviews.

### Model Structure

The first layer is the Embedded layer that uses 32 length vectors to represent each word.
 The next layer is the LSTM layer with 100 memory units (smart neurons). Finally, because this is a 
classification problem we use a Dense output layer with a single neuron and a sigmoid activation function
 to make 0 or 1 predictions for the two classes (good and bad) in the problem.

Because it is a binary classification problem, log loss is used as the loss function (binary_crossentropy in Keras).
 The efficient ADAM optimization algorithm is used. The model is fit for only 2 epochs because it quickly overfits the problem.
 A large batch size of 64 reviews is used to space out weight updates.

### Dropout

Recurrent Neural networks like LSTM generally have the problem of overfitting. Dropout can be applied
 between layers using the Dropout Keras layer. We can do this easily by adding new Dropout layers
 between the Embedding and LSTM layers and the LSTM and Dense output layers.

### LSTM and Convolutional Neural Network For Sequence Classification
Convolutional neural networks excel at learning the spatial structure in input data. The IMDB review
 data does have a one-dimensional spatial structure in the sequence of words in reviews and the CNN may
 be able to pick out invariant features for good and bad sentiment. This learned spatial features may
 then be learned as sequences by an LSTM layer.

We can easily add a one-dimensional CNN and max pooling layers after the Embedding layer which then
 feed the consolidated features to the LSTM. We can use a smallish set of 32 features with a small
 filter length of 3. The pooling layer can use the standard length of 2 to halve the feature map size.

