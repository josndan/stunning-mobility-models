# stunning-mobility-models

Worked with Akanksha Atrey and Mariem Snoussi to evaluate the effectiveness of traditional Machine Learning Models for Mobility prediction.

Prior work on personalized mobility modeling has focused on using Markov models and LSTMs. Identifying the effectiveness of different time-series modeling methods (e.g., hidden Markov models) for personalized mobility prediction can provide relevant insights for building robust personalized ML models. 

This project was done to evaluate the following hypothesis:
*Deep learning approaches for personalized mobility modeling are less effective than traditional ML methods.*

The following model were on mobility data 
**HMM 
Naive Bayes
Logistic Regression**

# Results:

|    Method   | Test accuracy |
|:-----------:|:-------------:|
|     LSTM    |     44.35%    |
|     HMM     |      35%      |
| Naive Bayes |     44.7%     |
|     SVM     |     44.89%    |

Traditional ML models are atleast as good as deep learning models such as LSTM for given mobility data.
