## tensorflow seq2seq chatbot

Build a general-purpose conversational chatbot based on a hot 
seq2seq approach implemented in [tensorflow](https://www.tensorflow.org/versions/master/tutorials/seq2seq/index.html#sequence-to-sequence_basics).

The current results are pretty lousy:

    hello baby	        - hello
    how old are you ?   - twenty .
    i am lonely	        - i am not
    nice                - you ' re not going to be okay .
    so rude	            - i ' m sorry .
    
Disclaimer: 

* the answers are hand-picked (it looks cooler that way)
* chatbot has no power to follow the conversation line so far; in the example above it's a just a coincidence (hand-picked one)

Everyone is welcome to investigate the code and suggest the improvements.

**Actual deeds**

* realise how to diversify chatbot answers (currently the most probable one is picked and it's dull)
* write the instructions to run the code on multple GPU's

We are trying to borrow the mentioned features from this awesome repo: 
[LeavesBreathe/Seq2Seq_Upgrade_TensorFlow](https://github.com/LeavesBreathe/Seq2Seq_Upgrade_TensorFlow).
Check it up.

**Papers**

* [Sequence to Sequence Learning with Neural Networks](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)
* [A Neural Conversational Model](http://arxiv.org/pdf/1506.05869v1.pdf)

**Nice picture**

[![seq2seq](https://4.bp.blogspot.com/-aArS0l1pjHQ/Vjj71pKAaEI/AAAAAAAAAxE/Nvy1FSbD_Vs/s640/2TFstaticgraphic_alt-01.png)](http://4.bp.blogspot.com/-aArS0l1pjHQ/Vjj71pKAaEI/AAAAAAAAAxE/Nvy1FSbD_Vs/s1600/2TFstaticgraphic_alt-01.png)


**Setup**

    git clone git@github.com:nicolas-ivanov/tf_seq2seq_chatbot.git
    cd tf_seq2seq_chatbot
    bash bin/setup.sh
    
**Run**

Train a seq2seq model on a small (17 MB) corpus of movie subtitles:

    python bin/train.py
    
(this command will run the training on a CPU... GPU instructions are coming)

Test trained trained model on a set of common questions:

    python bin/test.py
    
Chat with trained model in console:

    python bin/chat.py
    
All configuration params are stored at `tf_seq2seq_chatbot/configs/config.py`

**Requirements**

* [tensorflow](https://www.tensorflow.org/versions/master/get_started/os_setup.html)
* [Seq2Seq_Upgrade_TensorFlow](https://github.com/LeavesBreathe/Seq2Seq_Upgrade_TensorFlow) (`bash bin/setup.sh` gonna get it for you)
    