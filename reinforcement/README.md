Berkley Ai Reinforcement learning project.
All the codes have been written, you can just test them by running the 
commands.

#1 Value iteration on gridworld
python gridworld.py -a value -i 100 -k 10

#2 Crawler using q-learning
python crawler.py


#3 pacman small using q-learning
python pacman.py -p PacmanQAgent -x 2000 -n 2010 -l smallGrid 


#pacman large using approximate q-learning
python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 
-n 60 -l mediumClassic 
(This may take few minutes to train)
