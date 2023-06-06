# 224U-Project

**Research Question**
Study the gender-bias information distribution in mini-Bert layers

**Analysis Details**
1. Finetune mini-Bert on dataset with stereotypical pronouns (`S-Model`) and anti-stereotypical pronouns (`A-Model`) respectively
    - ❓Finetuning task: Predict Masked words
2. For each layers in the mini-Bert architecture:
    - Swap corresponding layer pairs from S-Model and A-Model 
    - Compute the percentage change after swapping --> a larger percentage change indicates a more concentrated gender-bias 
