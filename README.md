# Speaker-Identification-on-king-corpus using Feed Forward Networks
This project describes a feed forward network-based speaker recognition system in which a multi-layered neural network model is constructed for each speaker. 
The neural network models are trained using feature vectors obtained after speech processing. For testing the model, for each session, we read in all the data 
for that session and predict all frames. Then we fuse the scores by summing the log probabilities and decide class for that session.
