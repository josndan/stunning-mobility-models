import matplotlib.pyplot as plt
from dataloader import *
import pandas as pd
import numpy as np
from hmmlearn import hmm
from hmm import *

def experiment(x):
  evaluator = Metrics()
  n_components = [1,3,5,10,20,30]
  split_ratios =[0.8] #[0.1,0.3,0.5,0.8,0.9]
  lookbacks = [5] #[1,5,10,,15,20]
  m_accuracies = []
  m_recalls = []
  m_precisions = []
  m_f1_scores = []
  for n in n_components:
    for r in split_ratios:
        for l in lookbacks:
            train,test_X,test_y = test_train_split(x,lookback = l,ratio=r)
            hmms=[]
            for i in range(len(train)):
                hmm = HMM(type = "MHMM",n_components=n, n_iter=100)
                hmm.train(train[i])
                hmms.append(hmm)
            for i in range(len(train)):
                prediction = hmms[i].predict(test_X[i])
                acc = evaluator.accuracy(prediction,test_y[i])
                rec = evaluator.recall(prediction,test_y[i])
                pre = evaluator.precision(prediction,test_y[i])
                f1 = evaluator.f1_score(prediction,test_y[i])
                m_accuracies.append(acc)
                m_recalls.append(rec)
                m_precisions.append(pre)
                m_f1_scores.append(f1)

    print("n_component ",n)
    print("Avg. Accuracy ", np.average(m_accuracies)*100)
    print("Avg. Recall ", np.average(m_recalls))
    print("Avg. Precision ", np.average(m_precisions))
    print("Avg. F-1 Score ", np.average(m_f1_scores))
    print()

  m_accuracies = np.asarray(m_accuracies)
  m_recalls = np.asarray(m_recalls)
  m_precisions = np.asarray(m_precisions)
  m_f1_scores = np.asarray(m_f1_scores)
  fig, axs = plt.subplots(4,sharex=True)
  fig.suptitle('Metrics')
  axs[0].plot(n_components, m_accuracies)
  axs[1].plot(n_components, m_recalls)
  axs[2].plot(n_components, m_precisions)

  axs[3].plot(n_components, m_f1_scores)
  best_n = n_components[np.argmax(m_accuracies)]
  print("Best n : ", best_n)
  print("Best acc: ", np.max(m_accuracies)*100)

  axs[3].set_xlabel("n_component")
  axs[3].set_ylabel("F-1 Score")
  axs[2].set_ylabel("Precision")
  axs[1].set_ylabel("Recall")
  axs[0].set_ylabel("Accuracy")
  