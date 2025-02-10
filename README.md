## Experiment with Attention Mechanisms

| Attentions | Training Loss | Traning PPL | *Validation Loss | Validation PPL |
|----------|----------|----------|----------|----------|
|General Attention    | 3.306     | 27.270     | 3.947     | 51.755     |
| Multiplicative Attention   | 3.710     | 40.863     | 3.937     | 51.271     |
| Additive Attention  | 2.570     | 13.064     | 3.904     | 49.624     |

*These parameters are related to the lowest Validation loss during the traning.
## Evaluation and Verification
| General Attention | Multiplicative Attention | Additive Attention |
|------------------------|------------------------|------------------------|
|BLEU socre: 4.99 |BLEU socre: 4.03 |BLEU socre: 3.13|
|TEST PPL: 50.646 |TEST PPL: 50.770 |TEST PPL: 47.084|
| ![Image 1](genaralAttention1.png) | ![Image 2](multiplicativeAttention1.png) | ![Image 3](AdditiveAttention1.png) |
| ![Image 4](genaralAttention2.png) | ![Image 5](multiplicativeAttention2.png) | ![Image 6](AdditiveAttention2.png) |

## Analysis and Discussion

General Attention performs best in translation quality (BLEU), likely due to its ability to capture contextual relationships. But, it has higher perplexity, suggesting weaker probabilistic calibration.

Additive Attention good in training efficiency and generalization (lowest losses/PPL) but underperforms in BLEU, possibly due to overfitting.

Selecting Best model for machine translation:

Prioritize BLEU -> General Attention.

Prioritize training speed and perplexity -> Additive Attention.

General Attention is more effective for translation tasks, while Additive Attention is better for efficient training and sequence prediction.Thus, select Genaral attention for the Machine Translation.
