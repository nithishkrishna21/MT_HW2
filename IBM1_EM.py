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

sys.stderr.write("Training with EM's Algorithm...")
bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))][:opts.num_sents]
f_count = defaultdict(int)
e_count = defaultdict(int)
fe_count = defaultdict(int)
for (n, (f, e)) in enumerate(bitext):
  for f_i in set(f):
    f_count[f_i] += 1
    for e_j in set(e):
      fe_count[(f_i,e_j)] += 1
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
p_theta = {(f_i, e_j) : theta0 for (f_i, e_j) in fe_count.keys()}

while(k < opts.n_iters):
  k += 1
#   sys.stderr.write("Iteration : ")
  for (n, (f, e)) in enumerate(bitext):
    for (i, f_i) in enumerate(f): # go through all french words of sentence f[n]
      Z = 0
      for (j, e_j) in enumerate(e): # go through all english words of sentence e[n]
        Z += p_theta[(f_i, e_j)]
      for (j, e_j) in enumerate(e):
        c = p_theta[(f_i, e_j)] / Z
        fe_count[(f_i, e_j)] += c
        e_count[e_j] += c
    
  for (f_i, e_j) in p_theta.keys():
    p_theta[(f_i, e_j)] = fe_count[(f_i, e_j)] / e_count[e_j]


# Decode algorithm
for (n, (f, e)) in enumerate(bitext):
  for (i, f_i) in enumerate(f):
    best_prob = 0 # keeps track of the best alignment probability
    best_j = 0 # keeps track of the best alignment
    for (j, e_j) in enumerate(e):
      if p_theta[(f_i, e_j)] > best_prob:
        best_prob = p_theta[(f_i, e_j)]
        best_j = j

    sys.stdout.write("%i-%i " % (i,best_j))
  sys.stdout.write("\n")