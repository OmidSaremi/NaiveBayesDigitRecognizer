#!/usr/bin/env python

# The MIT License (MIT) Copyright (c) 2013 Omid Saremi 

from numpy import *  
import math
from  TrainingTestSets import *

# Read the digit data file into content
# For more info about structure of the input data, refer to "semeion.names.txt". Data was taken from UCI Machine learning repo: http://archive.ics.uci.edu/ml/datasets/Semeion+Handwritten+Digit

with open('digits.txt', 'r') as f: 
  content=f.readlines()

S=zeros((10, 256))  
p0=zeros((10, 256))
p1=zeros((10, 256))
count=zeros(10)
pClass=zeros(10)

# Labels the dataset entries as either training or test

print " "
print "What fraction of data to learn from? A number in (0, 1) : "

s=raw_input()

trainingSetLabels, testSetLabels=training_test_sets(float(s))
numTrainingEx=len(trainingSetLabels)

# Learning phase

for i in trainingSetLabels:
    l=content[i].rstrip().split(' ')
    l=[int(float(k)) for k in l]
    for j in range(0, 10):
       if l[256+j]==1:
             g=l[0:256]
             g.append(j) 
             g=array(g)
    label=g[256]
    S[label]=S[label]+g[0:256]
    count[label]=count[label]+1
    
# Computing the probability of a traing example belonging to one of the classes labeled as 0, 1, 2, 3, 4, 5, 6, 7, 8, 9

for i in range(0, 10):
  pClass[i]=count[i]/numTrainingEx

# Computing conditional probabilities: The probablity for given an image is in class "i", to have observed a given feature vector (represented by a 256 element pixel value vector)  

for i in range(0, 10): 
  for j in range(0, 256):
    p1[i][j]=S[i][j]/count[i]
    if(1.0-p1[i][j]<1e-10):
       p1[i][j]=0.999
       p0[i][j]=0.001
       continue
    elif (p1[i][j]<1e-10):
       p1[i][j]=0.001
       p0[i][j]=0.999
       continue
    else:   
       p0[i][j]=1.0-p1[i][j]

print " "
print "The End of Learning phase!" 
print " "
try:
    input("Press Enter to continue . . .")
except SyntaxError:
    pass
print " "    

# Now feed in the test set to measure the performance. Initializing the error rate

er=0  
for testEx in testSetLabels:
  test=[int(float(i))  for i in content[testEx].rstrip().split(' ')]
  dig=0
  for i in range(0, 10):
   if test[i+256]==1:
     print " "
     print "{0}\t{1}".format("The digit has been labeled as :", i) 
     dig=i
      
# Using (Naive) Bayes rule, computes the probability associated to a class "i" given a feature vector "observed" in the test set. I use "log" of probabilities to prevent underflow
  
  lnCondProb=zeros(10)
  for i in range(0, 10):
   for j in range(0, 256):
     if test[j]==0 :
       lnCondProb[i]=lnCondProb[i]+math.log(p0[i][j])
     else:
       lnCondProb[i]=lnCondProb[i]+math.log(p1[i][j])

# Decide which class has the highest probability      

  maximumProb=max(lnCondProb)
  index=0

  for i in range(0, 10): 
    if lnCondProb[i]==maximumProb:
       index=i
  print "{0}\t{1}".format("Naive Bayes prediction:", index)
  print " "
  
# If Naive Bayes algorithm's prediction is correct, print "I was right!" to the standard output, otherwise "My bad!"
  
  if (dig==index): 
    print "I was right!"
  else: 
    print "My bad!"
    er=er+1
  
# Print the error rate 

print er 
print len(testSetLabels)
print "{0} : {1} %".format("The error rate is ", 100*float(er)/len(testSetLabels))




