#!/usr/bin/env python
import optparse
import sys
from collections import defaultdict

optparser = optparse.OptionParser()
optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="Threshold for aligning with Dice's coefficient (default=0.5)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=100000000000, type="int", help="Number of sentences to use for training and alignment")
optparser.add_option("-i", "--iterations", dest="n_iters", default = 10, type = "int", help = "Number of iterations for EM")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (opts.train, opts.french)
e_data = "%s.%s" % (opts.train, opts.english)

sys.stderr.write("Training IBM 1 EM Algorithm...\n")
bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(e_data), open(f_data))][:opts.num_sents]
f_count = defaultdict(int)
e_count = defaultdict(int)
fe_count = defaultdict(int)
for (n, (e, f)) in enumerate(bitext):
  for f_i in set(f):
    f_count[f_i] += 1
    for e_j in set(e):
      fe_count[(e_j,f_i)] += 1
  for e_j in set(e):
    e_count[e_j] += 1
  if n % 500 == 0:
    sys.stderr.write(".")

# Implementation of EM Algorithm
k = 0
# initilaize thetas
theta0 = 0.01
p_theta = defaultdict(float)
# uniformly initialize the thetas
p_theta = {(e_j, f_i) : theta0 for (e_j, f_i) in fe_count.keys()}

while(k < opts.n_iters):
  k += 1
  sys.stderr.write(f" IBM 1 EM Iteration : {k}\n")
  for (n, (e, f)) in enumerate(bitext):
    for (j, e_j) in enumerate(e): # go through all english words of sentence e[n]
      s_total = defaultdict(float)
      for (i, f_i) in enumerate(f): # go through all french words of sentence f[n]
        s_total[e_j] += p_theta[(e_j, f_i)]
      for (i, f_i) in enumerate(f):
        c = p_theta[(e_j, f_i)] / s_total[e_j]
        fe_count[(e_j, f_i)] += c
        f_count[f_i] += c
    
  for (n, (e, f)) in enumerate(bitext):
    for (i, f_i) in enumerate(f):
      for (j, e_j) in enumerate(e):
        p_theta[(e_j, f_i)] = fe_count[(e_j, f_i)] / f_count[f_i]
sys.stderr.write("Finished training IBM 1 EM Algorithm.\n\n")

"""
# Decode algorithm
for (n, (e, f)) in enumerate(bitext):
  for (i, f_i) in enumerate(f):
    best_prob = 0 # keeps track of the best alignment probability
    best_j = 0 # keeps track of the best alignment
    for (j, e_j) in enumerate(e):
      if p_theta[(e_j, f_i)] > best_prob:
        best_prob = p_theta[(e_j, f_i)]
        best_j = j

    sys.stdout.write("%i-%i " % (i,best_j))
  sys.stdout.write("\n")
"""

sys.stderr.write("Training IBM 2 EM Algorithm...\n")
## Implementation of EM Algorithm for IBM 2 model

# carry over p_theta from IBM 1 model
# initialize a(i|j, le, lf) = 1/(lf + 1)
a = defaultdict(int)
for (n, (e, f)) in enumerate(bitext):
  for (j, e_j) in enumerate(e):
    l_e = len(e)
    for (i, f_i) in enumerate(f):
      l_f = len(f)
      a[(i, j, l_e, l_f)] = 1/(l_f + 1)

k = 0
while(k < opts.n_iters):

  k += 1
  sys.stderr.write(f"IBM 2 EM Iteration : {k}\n")
  # initialze
  fe_count = defaultdict(float)
  f_count = defaultdict(float)
  count_a = defaultdict(float)
  total_a = defaultdict(float)

  # for all sentence pairs e,f
  for (n, (e, f)) in enumerate(bitext):
    l_e, l_f = len(e), len(f)
    # compute normalization
    s_total = defaultdict(float)
    for (j, e_j) in enumerate(e):
      # s_total = defaultdict(float)
      for (i, f_i) in enumerate(f):
        s_total[e_j] += p_theta[(e_j, f_i)] * a[(i, j, l_e, l_f)]
      
    for (j, e_j) in enumerate(e):
      # collect counts
      for (i, f_i) in enumerate(f):
        c = (p_theta[(e_j, f_i)] * a[(i, j, l_e, l_f)]) / s_total[e_j]
        fe_count[(e_j, f_i)] += c
        f_count[f_i] += c
        count_a[(i, j, l_e, l_f)] += c
        total_a[(j, l_e, l_f)] += c
  
  for (e_j, f_i) in fe_count.keys():
    p_theta[(e_j, f_i)] = fe_count[(e_j, f_i)] / f_count[f_i]
  
  for (i, j, l_e, l_f) in count_a.keys():
    a[(i, j, l_e, l_f)] = count_a[(i, j, l_e, l_f)] / total_a[(j, l_e, l_f)]

sys.stderr.write("Finished Training IBM 2 EM ALgorithm.\n")    


# Decode algorithm
for (n, (e, f)) in enumerate(bitext):
  for (i, f_i) in enumerate(f):
    best_prob = 0 # keeps track of the best alignment probability
    best_j = 0 # keeps track of the best alignment
    l_f = len(f)
    for (j, e_j) in enumerate(e):
      l_e = len(e)
      prob = p_theta[(e_j, f_i)] * a[(i, j, l_e, l_f)]
      if prob > best_prob:
        best_prob = prob
        best_j = j

    sys.stdout.write("%i-%i " % (i,best_j))
  sys.stdout.write("\n")