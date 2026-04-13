"""
In the personaplex, we plot the log likelihood of that if the hidden is predicting (at step t) the next output (output of step t) (speaking_ll) and next input (input of step t+1)
Now we do the similar thing.
We read output_hidden under rootdir.We take the hidden state at step t and calculate the kl divergence between the distribution of the hidden state at the last layer (of text transformer) at t (server as the otuput reference) and the input embedding distribution at step t+1 (server as the input reference) for each layer in text tranformer


And we will read the input_timing like in the personaplex, and plot the kl divergence of each layer (as a heatmap, y is layer) and the middle of the picture should be anchor at the time indicated in the input timing. see how personaplex plot the CE heatmap.
So if there is two anchor point, we will have four heatmaps. (two (input, output) for each anchor point)

and other naming converntioini should be like the ones in personaplex.
"""
