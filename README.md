# Othello-Q-Learning

To run the program, first ensure that you have the correct libraries installed. This can be done with the following command:

```console
pip install -r requirements.txt
```

Then, to run the program, run the following command:

```console
python DeepQ_learning.py
```

The program will run training for 1,0000 episodes and then run through 1,000 test games. This whole process takes around 90 seconds.

The code for my Othello enviornment can be found in `othello.py`
The code for the Deep Q-Learning agent can be found in `DeepQ_learning.py`
the code for the Q Network can be found in `QNetwork.py`

The figures used in my report can be found in the `images` folder.
The initial Q-learning project can be found in the `initial_q_learning` folder, the code in there is very messy. It wasn't until I started the Deep Q-Learning project that I started to clean up my code.
