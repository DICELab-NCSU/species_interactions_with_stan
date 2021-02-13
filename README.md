# Species interaction models with Stan (and R)
This collection contains code for fitting common species interaction models using Hamiltonian Monte Carlo in Stan. It is intended as a learning resource, rather than flexible "production-ready" implementations. Expect that the code may contain errors and inefficiencies--use at your own risk!

Each model is contained in a sub-folder embedded in a GitHub-viewable Markdown document (`.md`), an RMarkdown document (`.Rmd`), and a stand-alone Stan file (`.stan`).

## Models available
### Competition
- [ ] Lotka-Volterra (discrete time)
- [ ] Lotka-Volterra (continuous time)
- [x] Beverton-Holt
- [ ] Leslie-Gower
- [ ] Law-Wilkinson
- [ ] Ricker
- [ ] Hassell

### Consumer-Resource
- [ ] Lotka-Volterra (discrete time)
- [ ] Lotka-Volterra (continuous time)
- [ ] Rosenzweig-MacArthur
- [ ] Beverton-Holt
- [ ] Leslie-Gower
- [ ] Nicholson-Bailey
- [ ] Arditi-Ginzburg
- [ ] Yodzis-Innes
- [ ] DeAngelis-Beddington
- [ ] Wangersky-Cunningham

## Contributing
We welcome contributions to this example model base. Please use `model_template.Rmd` in the home directory to format your contribution. Please include the following in your pull request:
- an RMarkdown source file
- a rendered Markdown file with `*_files` subfolder (e.g., containing images of plots)
- a stand-alone Stan file

We also welcome bug reports & ideas to make the models more efficient.

Please note our [code of conduct for contributors](CONDUCT.md).
