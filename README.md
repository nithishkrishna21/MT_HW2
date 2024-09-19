HW2 Group - Nithish Krishna Shreenevasan (nshreen1@jh.edu), Khushang Zaveri (kzaveri1@jh.edu), Avantika Singh (asing153@jh.edu)


We have implemented the IBM1 model and trained it using EM (File name: IBM1_EM.py)
We have implemented the IBM 2 Model as our additional Model (File name : IBM2_EM.py)

How to run the files:

1) To run the file and get alignments:

	python file.py -n num_sentences -i num_iterations > alignment_file.a

num_sentences -> [1, 10000]
num_iterations -> can be run for any number of times but it is recommended to run for 5 iterations

2) To score the alignments:
	
	python score-alignments < alignment_file.a

Our EM_IBM1_alignment.a file are the alignments obtained after training IBM 1 model using EM algorithm
Our alignment.a file are the alignments obtained after training IBM 2 model using EM algorithm 

We have included screenshots of the results for your ready reference. 

All files including the report, code, pictures, etc., are included in the zip file
