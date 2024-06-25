# SOC24Midterm

Welcome to my midterm report of SOC 2024, GPT Mastery.

Let me describe the files present in this repo.

```the-verdict.txt``` and ```wizard_of_oz.txt``` are the text files on which we intend to train our model.

I first started with learning how to use PyTorch, which can be seen in ```torchtrials.py```.

Then the ```bigram.py``` file consists of the basic stuff like creating my own tokenizer and just splitting the data into train, test and validation sets.

Next the ```book_impl.py``` file has a slightly better tokenizer and also contains the _Dataset class_ and the _Data loader_. It also contains a test embedding layer. 

In the ```attention.py``` file, I implemented the _Simple Attention_, _Causal Attention_ and the _Multihead Attention_ classes.

The ```transformer.py``` file consists of the transformer block which uses multiple layers(_normalization_, _GELU_ and _feedforward_) which are implemented in ```gpt2.py```.
```gpt2.py``` also consists of the final model(yet to be pre trained) formed from using the transformer block multiple times and a final normalization layer.

I used the ```pretrainedgpt2.py``` and ```gpt_download.py``` files to download the pre trained weights of GPT2 from OpenAI and ```rnad1.py``` file to implement my own GPT using the pre trained weights.

The ```pretraining.py``` file consists of manual pre training of my own GPT using the text files I downloaded from the internet. This part is not yet complete.

The remaining part of the project is to pre train the model first and then fine tune it to answer our questions.
