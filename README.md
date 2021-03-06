# Wasserstein Generative Adversarial Uncertainty Quantification in Physics-Informed Neural Networks
### Yihang Gao and Michael K. Ng
In this paper, we study a physics-informed algorithm for Wasserstein Generative Adversarial Networks (WGANs) for uncertainty quantification in solutions of partial differential equations. By using groupsort activitation functions in adversarial network discriminators, network generators are utilized to learn the uncertainty in solutions of partial differential equations observed from the initial/boundary data. Under mild assumptions, we show that the generalization error of the computed generator converges to the approximation error of the network with high probability, when the number of samples are sufficiently taken. According to our established error bound, we also find that our physics-informed WGANs have higher requirement for the capacity of discriminators than that of generators. Numerical results on synthetic examples of partial differential equations are reported to validate our theoreical results and demonstrate how uncertainty quantification can be obtained for solutions of partial differential equations and the distributions of initial/boundary data. 


#### For more details, please refer to the [paper](https://arxiv.org/abs/2108.13054) and the tutorial codes (.ipynb file) of four examples.


### Citation
```
@article{gao2021wasserstein,
  title={Wasserstein Generative Adversarial Uncertainty Quantification in Physics-Informed Neural Networks},
  author={Gao, Yihang and Ng, Michael K},
  journal={arXiv preprint arXiv:2108.13054},
  year={2021}
}
```
