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
1. (DONE)Switch all attention weights from anti-model to the vanilla model
2. (DONE)Change only partial weights from individual attention blocks
3. (DONE)Finetune using full dataset
4. (probably skip - discussion)hyperparameter tuning?
5. BERT of another size: analysis only
6. Finetuning on a large amount of anti-stereo data seemed to teach the model to learn to predict anti-stereo answers. May consider train on **balanced dataset** (including anti & stereo).

**Analysis TODO**
1. compare finetuend bert result with vanilla bert result: if number of anti-stereotyped prediction increase, compare probabilities

**Analysis Result**
1. vanilla BERT is biased in its prediction
2. finetuning vanilla BERT with large anti-stereo dataset has noticeable impact on its predictions.
3. Full dataset contains much more stereo data, which might be a reflection on the dataset used to pretrain existing models


