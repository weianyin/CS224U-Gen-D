# 224U-Project

**Research Question**
Study the gender-bias information distribution in mini-Bert layers

**Analysis Details**
1. Finetune mini-Bert on dataset with stereotypical pronouns (`S-Model`) and anti-stereotypical pronouns (`A-Model`) respectively
    - Finetuning task: Predict Masked words
2. For each layers in the mini-Bert architecture:
    - Swap corresponding layer pairs from S-Model and A-Model 
    - Compute the percentage change after swapping --> a larger percentage change indicates a more concentrated gender-bias 

**Limitations**
1. BERT cannot predict pronouns 
2. We only considered "he" and "she", but a desired prediction is "them" 
3. Does not detect/consider implicit bias
4. 
**TODO**
1. (Done)Switch all attention weights from anti-model to the vanilla model
2. Change only partial weights from individual attention blocks
3. (DONE)Fine tune using full dataset
4. hyperparameter tuning?
5. BERT of another size
