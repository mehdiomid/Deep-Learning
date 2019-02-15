# LSTM Sentiment Analysis of a Tweet

Here is to identify and categorize opinions expressed in a piece of text,
 especially in order to determine whether the writer's attitude towards a particular topic,
 product is positive, negative, or neutral.

## Data

The dataset is obtained from kaggle:

First GOP Debate Twitter Sentiment: Analyze tweets on the first 2016 GOP Presidential Debat

https://www.kaggle.com/crowdflower/first-gop-debate-twitter-sentiment/version/2

## Method

A Recurrent Neural Network, LSTM is used to train a model that will be used to predict the sentiment of the tweets 

To make this a binary clasification, the 'Neutral' sentiments are being dropped.
 As a result we only differentiate positive and negative tweets. It is also necessary to
clean the texts to have only valid characters and words. We use Tokenizer to vectorize and convert
 text into Sequences so the Network can deal with it as input.

The number of max features:2000

In the LSTM Network, embed_dim, lstm_out, batch_size, droupout_x variables are hyperparameters and shall be tuned.
 Also a softmax is used as activation function. The reason is that the Network is using categorical
 crossentropy, and softmax is just the right activation method for that.

Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 28, 128)           256000    
_________________________________________________________________
spatial_dropout1d_1 (Spatial (None, 28, 128)           0         
_________________________________________________________________
lstm_1 (LSTM)                (None, 196)               254800    
_________________________________________________________________
dense_1 (Dense)              (None, 2)                 394       
=================================================================
Total params: 511,194
Trainable params: 511,194
Non-trainable params: 0
_________________________________________________________________




 