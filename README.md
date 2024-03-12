# Gen-D: Analysis Framework for Gender-specific Information Distr

## Research Question

Social biases in language models have widely raised public attention. Current mitigations in- volve correcting biases in word embeddings, generating gender-neutral datasets, and debiasing models as a whole. Each mitigation comes with drawbacks – word embeddings fall short of ensuring sentence-level fairness, and debiasing models as a whole using gender- neutral datasets induces burden on data col- lection. 

In this paper, we propose an analysis framework to characterize gender-specific information distribution for general language models. Our findings include that **fine-tuning attention layers only**, instead of the whole model, using anti-bias datasets is enough for debiasing, and gender-specific information is concentrated in a small portion of model parameters. Our proposed method has great potential in reducing cost for collecting large gender- neutral data and providing insights for effective model debiasing by targeting specific layers.

## Models Evaluted

- BERT (bert-base-uncased)
- DistillBERT (distilbert-base-uncased)


## Methodologies

### Bias Evaluation

