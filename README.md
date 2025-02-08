## Experiment with Attention Mechanisms

| Attentions | Training Loss | Traning PPL | Validation Loss | Validation PPL |
|----------|----------|----------|----------|----------|
|General Attention    | 3.815     | 45.379     | 4.111     | 61.010     |
| Multiplicative Attention   | 3.383     | 29.473     | 3.942     | 51.510     |
| Additive Attention  | 2.987     | 19.835     | 3.849     | 46.941     |

This experiment was conducted with RNN /GRU + Attention 
## Evaluation and Verification
| General Attention | Multiplicative Attention | Additive Attention |
|------------------------|------------------------|------------------------|
|Inferance time: 82.65s |Inferance time: 136.12s |Inferance time: 134.61s |
|BLEU socre: 1.45 |BLEU socre:  |BLEU socre: 1.22|
| ![Image 1](genaralAttention1.png) | ![Image 2](multiplicativeAttention1.png) | ![Image 3](AdditiveAttention1.png) |
| ![Image 4](genaralAttention2.png) | ![Image 5](multiplicativeAttention2.png) | ![Image 6](AdditiveAttention2.png) |
