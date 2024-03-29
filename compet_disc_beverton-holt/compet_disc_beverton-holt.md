Competitive Beverton-Holt in discrete time
================
W.K. Petry
2021-Sep-27

## The model

<table>
<colgroup>
<col style="width: 31%" />
<col style="width: 68%" />
</colgroup>
<tbody>
<tr class="odd">
<td><strong>Number of species</strong></td>
<td>1</td>
</tr>
<tr class="even">
<td><strong>Time</strong></td>
<td>discrete</td>
</tr>
<tr class="odd">
<td><strong>Interaction type(s)</strong></td>
<td>competition</td>
</tr>
<tr class="even">
<td><strong>Difference equation</strong></td>
<td><span class="math inline"><em>n</em><sub><em>t</em> + 1</sub> = <em>n</em><sub><em>t</em></sub><em>λ</em>/(1+<em>α</em><em>n</em><sub><em>t</em></sub>)</span></td>
</tr>
<tr class="odd">
<td><strong>Parameter meaning</strong></td>
<td><p><span class="math inline"><em>n</em><sub><em>t</em></sub></span>: number of individuals at time <span class="math inline"><em>t</em></span></p>
<p><span class="math inline"><em>λ</em></span>: per-capita intrinsic rate of increase</p>
<p><span class="math inline"><em>α</em></span>: (intra-specific) competition coefficient</p></td>
</tr>
<tr class="even">
<td><strong>Equilibrium</strong></td>
<td><span class="math inline"><em>n</em><sup>*</sup> = (<em>λ</em>−1)/<em>α</em></span></td>
</tr>
<tr class="odd">
<td><strong>Original publication</strong></td>
<td><span class="citation" data-cites="beverton1957">Beverton and Holt (1957)</span>, using carrying capacity parameterization</td>
</tr>
</tbody>
</table>

## Bayesian hierarchical model structure

This framing of the model imagines that you have individual-level data
on focal individuals. that includes the number of neighboring
competitors (*n*<sub>*t*</sub>) and the number of offspring produced
(*n*<sub>off, *t*</sub>). The observation of the latent individual
fecundities (*f*<sub>*t*</sub>) is a Poisson process. Domain expertise
is used to set the priors on *λ* and *α*.

$$
\\begin{aligned}
n\_{\\mathrm{off}, t} &\\sim \\mathrm{Poisson}\\left(f\_{t}\\right) \\\\
f_t &= \\frac{\\lambda}{1+\\alpha n\_{t}} \\\\
\\lambda &\\sim \\mathrm{LogNormal}\\left(\\mu\_{\\lambda},\\sigma\_{\\lambda}\\right) \\\\
\\alpha &\\sim \\mathrm{HalfNormal}\\left(\\mu\_{\\alpha},\\sigma\_{\\alpha}\\right).
\\end{aligned}
$$

(If this equation looks jumbled, you may need to download the source
.Rmd or use a [browser
add-on](https://chrome.google.com/webstore/detail/mathjax-plugin-for-github/ioemnmodlmafdkllaclgeombjnmnbima)
to render it.)

## Stan code

``` r
## prepare Rmarkdown to input model
library(cmdstanr)  # remotes::install_github("stan-dev/cmdstanr")
register_knitr_engine()
```

``` stan
// Competitive Beverton-Holt in discrete time
// W.K. Petry

// define the structure & types of data to which the model is fit
data {
  int<lower=0> N;                 // number of focal individuals (data rows)
  int<lower=0> offspring[N];      // observed offspring production by focals
  int<lower=0> bkgd_dens[N];      // local densities around focal individuals
  // accept priors from user
  real<lower=0> lambda_mean;      // prior mean for lambda
  real<lower=0> lambda_sd;        // prior standard deviation for lambda
  real<lower=0> alpha_mean;       // prior mean for competition coefficient
  real<lower=0> alpha_sd;         // prior standard deviation for competition coefficent
}

// define the parameters to be fit
parameters {
  real<lower=0> lambda;          // per-capita intrinsic rate of increase
  real<lower=0> alpha;           // competition coefficient
}

transformed parameters {
  vector[N] fec;                 // turn data-generating process crank
  for (i in 1:N) {               // to get latent fecundity
    fec[i] = lambda / (1. + alpha * bkgd_dens[i]);
  }
}

model {
  offspring ~ poisson(fec);     // observe latent fecundity as whole offspring
  lambda ~ lognormal(lambda_mean, lambda_sd);  // lognormal prior on lambda
  alpha ~ normal(alpha_mean, alpha_sd);        // (half)normal prior on alpha
}
```

## Example data & fitting

``` r
library(posterior)
library(ggplot2)

## set 'true' parameter values
lambda <- 220
alpha <- 1.25

## generate synthetic data
set.seed(58428)
N <- 100L  # number of observed individuals
bkgd_dens <- as.integer(round(runif(n = N, min = 0L, max = 250L)))
offspring <- rpois(n = N, lambda = (lambda / (1 + alpha * bkgd_dens)))

dat <- list(N = N, bkgd_dens = bkgd_dens, offspring = offspring)

## add priors
## (note both parameters are truncated at zero by the Stan code)
dat$lambda_mean <- 200
dat$lambda_sd <- 30
dat$alpha_mean <- 0.5
dat$alpha_sd <- 0.75

## compile the model from an external .stan file
bh <- cmdstan_model("compet_disc_beverton-holt.stan")

## fit the model (sample the posterior)
ps <- bh$sample(data = dat, iter_warmup = 1000, iter_sampling = 2000, refresh = 1000)
```

    ## Running MCMC with 4 sequential chains...
    ## 
    ## Chain 1 Iteration:    1 / 3000 [  0%]  (Warmup) 
    ## Chain 1 Iteration: 1000 / 3000 [ 33%]  (Warmup) 
    ## Chain 1 Iteration: 1001 / 3000 [ 33%]  (Sampling) 
    ## Chain 1 Iteration: 2000 / 3000 [ 66%]  (Sampling) 
    ## Chain 1 Iteration: 3000 / 3000 [100%]  (Sampling) 
    ## Chain 1 finished in 0.5 seconds.
    ## Chain 2 Iteration:    1 / 3000 [  0%]  (Warmup) 
    ## Chain 2 Iteration: 1000 / 3000 [ 33%]  (Warmup) 
    ## Chain 2 Iteration: 1001 / 3000 [ 33%]  (Sampling) 
    ## Chain 2 Iteration: 2000 / 3000 [ 66%]  (Sampling) 
    ## Chain 2 Iteration: 3000 / 3000 [100%]  (Sampling) 
    ## Chain 2 finished in 0.3 seconds.
    ## Chain 3 Iteration:    1 / 3000 [  0%]  (Warmup) 
    ## Chain 3 Iteration: 1000 / 3000 [ 33%]  (Warmup) 
    ## Chain 3 Iteration: 1001 / 3000 [ 33%]  (Sampling) 
    ## Chain 3 Iteration: 2000 / 3000 [ 66%]  (Sampling) 
    ## Chain 3 Iteration: 3000 / 3000 [100%]  (Sampling) 
    ## Chain 3 finished in 0.4 seconds.
    ## Chain 4 Iteration:    1 / 3000 [  0%]  (Warmup) 
    ## Chain 4 Iteration: 1000 / 3000 [ 33%]  (Warmup) 
    ## Chain 4 Iteration: 1001 / 3000 [ 33%]  (Sampling) 
    ## Chain 4 Iteration: 2000 / 3000 [ 66%]  (Sampling) 
    ## Chain 4 Iteration: 3000 / 3000 [100%]  (Sampling) 
    ## Chain 4 finished in 0.4 seconds.
    ## 
    ## All 4 chains finished successfully.
    ## Mean chain execution time: 0.4 seconds.
    ## Total execution time: 2.1 seconds.

``` r
pdat <- as_draws_df(ps$draws(variables = c("lambda", "alpha")))

## examine posterior distribution vs. true values
c(lambda = lambda, alpha = alpha)  # true values
```

    ## lambda  alpha 
    ## 220.00   1.25

``` r
summarize_draws(ps$draws(variables = c("lambda", "alpha")))  # posterior summary
```

    ## # A tibble: 2 × 10
    ##   variable   mean median     sd    mad      q5    q95  rhat ess_bulk ess_tail
    ##   <chr>     <dbl>  <dbl>  <dbl>  <dbl>   <dbl>  <dbl> <dbl>    <dbl>    <dbl>
    ## 1 lambda   200.   200.   27.5   27.5   154.    245.    1.00    1867.    1850.
    ## 2 alpha      1.15   1.14  0.176  0.178   0.859   1.44  1.00    1835.    2018.

``` r
ggplot(data = NULL, aes(x = alpha, y = lambda))+
  geom_hex(data = pdat)+
  geom_point(aes(color = "true value"), shape = 3, stroke = 2, size = 4)+
  scale_color_manual(name = "", values = "black")+
  scale_fill_distiller(name = "density", palette = "RdPu",
                       direction = 1)+
  theme_minimal(base_size = 20)+
  ggtitle("Joint posterior distribution")
```

![](compet_disc_beverton-holt_files/figure-gfm/example-1.png)<!-- -->

``` r
## calculate the posterior equilibrium population size, n^*
nstar <- (pdat$lambda - 1) / pdat$alpha
ggplot(data = NULL, aes(x = nstar))+
  geom_density(fill = "cornflowerblue", size = 0)+
  geom_vline(xintercept = (lambda - 1) / alpha, size = 1)+  # true equilibrium
  labs(title = "Posterior distribution of n*", x = "n*")+
  theme_minimal(base_size = 20)+
  theme(axis.text.y = element_blank())
```

![](compet_disc_beverton-holt_files/figure-gfm/example-2.png)<!-- -->

## Variations

-   Change the observation process to allow for overdispersion (e.g.,
    negative binomial)

-   Set heavier-tailed prior on *α* (e.g., Student’s t)

## Session info

``` r
sessionInfo()
```

    ## R version 4.1.0 (2021-05-18)
    ## Platform: x86_64-apple-darwin17.0 (64-bit)
    ## Running under: macOS Mojave 10.14.6
    ## 
    ## Matrix products: default
    ## BLAS:   /Library/Frameworks/R.framework/Versions/4.1/Resources/lib/libRblas.dylib
    ## LAPACK: /Library/Frameworks/R.framework/Versions/4.1/Resources/lib/libRlapack.dylib
    ## 
    ## locale:
    ## [1] en_US.UTF-8/en_US.UTF-8/en_US.UTF-8/C/en_US.UTF-8/en_US.UTF-8
    ## 
    ## attached base packages:
    ## [1] stats     graphics  grDevices utils     datasets  methods   base     
    ## 
    ## other attached packages:
    ## [1] ggplot2_3.3.5       posterior_1.1.0     cmdstanr_0.4.0.9000
    ## 
    ## loaded via a namespace (and not attached):
    ##  [1] tidyselect_1.1.1     xfun_0.26            purrr_0.3.4         
    ##  [4] lattice_0.20-45      colorspace_2.0-2     vctrs_0.3.8         
    ##  [7] generics_0.1.0       htmltools_0.5.2      yaml_2.2.1          
    ## [10] utf8_1.2.2           rlang_0.4.11         hexbin_1.28.2       
    ## [13] pillar_1.6.3         glue_1.4.2           withr_2.4.2         
    ## [16] DBI_1.1.1            RColorBrewer_1.1-2   distributional_0.2.2
    ## [19] matrixStats_0.61.0   lifecycle_1.0.1      stringr_1.4.0       
    ## [22] munsell_0.5.0        gtable_0.3.0         evaluate_0.14       
    ## [25] labeling_0.4.2       knitr_1.34           fastmap_1.1.0       
    ## [28] ps_1.6.0             fansi_0.5.0          highr_0.9           
    ## [31] scales_1.1.1         backports_1.2.1      checkmate_2.0.0     
    ## [34] jsonlite_1.7.2       abind_1.4-5          farver_2.1.0        
    ## [37] tensorA_0.36.2       digest_0.6.28        stringi_1.7.4       
    ## [40] processx_3.5.2       dplyr_1.0.7          grid_4.1.0          
    ## [43] cli_3.0.1            tools_4.1.0          magrittr_2.0.1      
    ## [46] tibble_3.1.4         crayon_1.4.1         pkgconfig_2.0.3     
    ## [49] ellipsis_0.3.2       data.table_1.14.0    assertthat_0.2.1    
    ## [52] rmarkdown_2.11       rstudioapi_0.13      R6_2.5.1            
    ## [55] compiler_4.1.0

## References

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-beverton1957" class="csl-entry">

Beverton, R.J., and S.J. Holt. 1957. *On the Dynamics of Exploited Fish
Populations*. *Fishery Investigations Series II*. Vol. XIX. Ministry of
Agriculture, Fisheries and Food.

</div>

<div id="ref-cushing2004jdiffeqappl" class="csl-entry">

Cushing, J.M., S. Leverge, N. Chitnis, and S.M. Henson. 2004. “Some
Discrete Competition Models and the Competitive Exclusion Principle.” *J
Diff Eq Appl* 10.

</div>

</div>
