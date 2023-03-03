# FrankenSplit
Repository for the paper on Saliency Guided Neural Feature Compression with Shallow Variational Bottleneck Injection 


~~I will push the code in the next 1-2 weeks , the repository I worked on became a mess and an amalgamy of 5 different projects.~~

**Update**: 
I'll be rather busy the next few weeks, so I've pushed majority of the code for the main experiment, but I've probably broken something in the process. Will make sure to add instructions on how to reproduce the results from the paper, upload some pretrained weights and make sure nothing is broken in the next two weeks

## Short Description
- Todo

## Setup
- Todo

## Notes
- I don't plan on actively maintaing this repository/monitor issues after I've pushed all the experiment code. In case you need my assitance, or you notice some problems (e.g. missed reference, broken implementation) please contact me at: alireza.furutanpey@dsg.tuwien.ac.at
- Check out [torchdistill](https://github.com/yoshitomo-matsubara/torchdistill) (It's awesome!) for documentation on how configurations are loaded and how you can adjust them if you want to perform your own experiments
- I've removed most code that was out of scope to include in the paper to avoid confusion but there are still some references to unpublished implementations/results

# Citation
[Preprint](https://arxiv.org/abs/2302.10681)
```bibtex
@article{furutanpey2023frankensplit,
      title={FrankenSplit: Saliency Guided Neural Feature Compression with Shallow Variational Bottleneck Injection}, 
      author={Alireza Furutanpey and Philipp Raith and Schahram Dustdar},
      year={2023},
      eprint={2302.10681},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```

# References
- Matsubara, Yoshitomo. "torchdistill: A modular, configuration-driven framework for knowledge distillation." Reproducible Research in Pattern Recognition: Third International Workshop, RRPR 2021, Virtual Event, January 11, 2021, Revised Selected Papers. Cham: Springer International Publishing, 2021.
- Matsubara, Yoshitomo, et al. "SC2: Supervised compression for split computing." arXiv preprint arXiv:2203.08875 (2022).
- Wightman, Ross. "Pytorch image models." (2019).
- Bégaint, Jean, et al. "Compressai: a pytorch library and evaluation platform for end-to-end compression research." arXiv preprint arXiv:2011.03029 (2020).
- Gildenblat, Jacob. "contributors. Pytorch library for cam methods." (2021).
- Ballé, Johannes, et al. "Variational image compression with a scale hyperprior." arXiv preprint arXiv:1802.01436 (2018).
- Minnen, David, Johannes Ballé, and George D. Toderici. "Joint autoregressive and hierarchical priors for learned image compression." Advances in neural information processing systems 31 (2018).
