# Load Disaggregation with Attention
This repository provides the implementation of LDwA (Load Disaggregation with Attention) described in the paper:

> V. Piccialli and A.M. Sudoso, Improving Non-Intrusive Load Disaggregation through an Attention-Based Deep Neural Network. 
> Energies 2021, 14, 847. https://doi.org/10.3390/en14040847

All code is written in Python using Tensorflow.

Citation export:

```
@Article{en14040847,
AUTHOR = {Piccialli, Veronica and Sudoso, Antonio M.},
TITLE = {Improving Non-Intrusive Load Disaggregation through an Attention-Based Deep Neural Network},
JOURNAL = {Energies},
VOLUME = {14},
YEAR = {2021},
NUMBER = {4},
ARTICLE-NUMBER = {847},
URL = {https://www.mdpi.com/1996-1073/14/4/847},
ISSN = {1996-1073},
ABSTRACT = {Energy disaggregation, known in the literature as Non-Intrusive Load Monitoring (NILM), is the task of inferring the power demand of the individual appliances given the aggregate power demand recorded by a single smart meter which monitors multiple appliances. In this paper, we propose a deep neural network that combines a regression subnetwork with a classification subnetwork for solving the NILM problem. Specifically, we improve the generalization capability of the overall architecture by including an encoder–decoder with a tailored attention mechanism in the regression subnetwork. The attention mechanism is inspired by the temporal attention that has been successfully applied in neural machine translation, text summarization, and speech recognition. The experiments conducted on two publicly available datasets—REDD and UK-DALE—show that our proposed deep neural network outperforms the state-of-the-art in all the considered experimental conditions. We also show that modeling attention translates into the network’s ability to correctly detect the turning on or off an appliance and to locate signal sections with high power consumption, which are of extreme interest in the field of energy disaggregation.},
DOI = {10.3390/en14040847}
}
```
