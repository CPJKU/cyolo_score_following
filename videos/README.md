### Videos of our best model (CYOLO-IR) trying to track several pieces from the test set

On the left you will see the score image together with the prediction of the model. (Note that the repetitions are not considered in this work)
To the right we visualize the last two seconds of audio. The prediction itself will be based on more than this excerpt (see. Section III).
The predicted bounding box is visualized in red and the actual ground truth position as a blue line.

In the folder `synth-test` you find two pieces which are rendered from the score MIDI with the same piano sound-font seen during training. 
The tracker is able to follow those pieces quite accurate, but it also makes some minor mistakes.

- Frédéric Chopin - O28-11 Prelude
- Pyotr Ilyich Tchaikovsky - O39-16 Old French Song
 
In the folder `real-test` you find three pieces with two different audio scenarios each, the direct out of the piano and a corresponding room recording.
One can observe that this "real-audio" setting is harder for the system to track, i.e. it is less accurate, compared to the synthetic setting, but overall it is never completely lost.

- Johann Sebastian Bach - BMV924a Prelude - Performed by Jan Hajič jr.
- Wolfgang Amadeus Mozart - KV331 Variation 1 - Performed by Carlos Eduardo Cancino Chacón
- Robert Schumann - O68-16 Premier Chagrin - Performed by Carlos Eduardo Cancino Chacón
