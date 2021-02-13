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
  lambda ~ normal(lambda_mean, lambda_sd);  // normal prior on lambda
  alpha ~ normal(alpha_mean, alpha_sd);     // normal prior on alpha
}
