# Species interaction models with Stan (and R)
This collection contains code for fitting common species interaction models using Hamiltonian Monte Carlo in Stan. It is intended as a learning resource, rather than flexible "production-ready" implementations. Expect that the code may contain errors and inefficiencies--use at your own risk!

Each model is contained in a sub-folder. The naming convention specifies each model as three components separated by underscores:
- the interaction type (`compet_*` for competitive, `conres_*` for consumer-resource)
- the time framework (`*_disc_*` for discrete, `*_cont_*` for continuous)
- the common name of the model in the ecological literature

Each folder has a GitHub-viewable Markdown document (`.md`), the RMarkdown document (`.Rmd`) that generated it, and a stand-alone Stan file (`.stan`) containing the portable model source code. [`R`](https://cran.r-project.org/) is used to either generate or import example data and interfaces Stan via the [`cmdstanr`](https://mc-stan.org/cmdstanr/) package. The posterior distribution of the model parameters are presented, and when possible a meaningful calculation on the posterior parameters is performed (e.g., finding the posterior distribution of the equilibrium point).

## Models available
### Competition (compet_*)
- [ ] Lotka-Volterra (discrete time)
- [ ] Lotka-Volterra (continuous time)
- [x] Beverton-Holt
- [ ] Leslie-Gower
- [ ] Law-Wilkinson
- [ ] Ricker
- [ ] Hassell

### Consumer-Resource (conres_*)
- [ ] Lotka-Volterra (discrete time)
- [x] Lotka-Volterra (continuous time)
- [ ] Rosenzweig-MacArthur
- [ ] Beverton-Holt
- [ ] Leslie-Gower
- [ ] Nicholson-Bailey
- [ ] Arditi-Ginzburg
- [ ] Yodzis-Innes
- [ ] DeAngelis-Beddington
- [ ] Wangersky-Cunningham

## Similar resources
[Stan ecology page](https://stanecology.github.io/): additional case studies, event materials, & collected peer-reviewed papers

[Spatial models for plant neighborhood dynamics in Stan](https://mc-stan.org/users/documentation/case-studies/plantInteractions.html): a spatially-explicit competition model wherein competitors affect focals more when they are larger and closer

[Ecology tag on Stan Discourse](https://discourse.mc-stan.org/tag/ecology)

## Contributing
We welcome contributions to this example model base. Please use `model_template.Rmd` in the home directory to format your contribution. Please include the following in your pull request:
- an RMarkdown source file
- a rendered Markdown file with `*_files` subfolder (e.g., containing images of plots)
- a stand-alone Stan file

We also welcome bug reports & ideas to make the models more efficient.

Please note our [code of conduct for contributors](CONDUCT.md).
