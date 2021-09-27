// Competitive n-species Leslie-Gower in discrete time
// interaction coefficients are regularized as the product of species competitive
// effects & responses, plus normal error
// W.K. Petry

// define the structure & types of data to which the model is fit
data {
  int<lower=0> N;                 // number of focal individuals (data rows)
  int<lower=0> nsp;               // number of species
  int<lower=0> focalsp[N];        // species index of the focal individual
  int<lower=0> offspring[N];      // observed offspring production by focal individual
  matrix[N,nsp] bkgd_dens;        // local densities by species (cols) around focal individuals (rows)
  // accept priors from user
  vector[nsp] lambda_mean;        // prior means for lambda
  vector[nsp] lambda_sd;          // prior standard deviations for lambda
  vector[nsp] es_mean;            // prior mean for competitive EFFECTS
  vector[nsp] es_sd;              // prior standard deviation for competitive EFFECTS
  vector[nsp] rs_mean;            // prior mean for competitive RESPONSES
  vector[nsp] rs_sd;              // prior standard deviation for competitive RESPONSES
  real<lower=0> alphasd_scale;    // cauchy scale parameter for sd of alpha_ij deviations from r_i*e_j
}

transformed data {
  int<lower=0> nalphas = nsp*nsp;   // number of coefficients in competition matrix
}

// define the parameters to be fit
parameters {
  vector<lower=0>[nsp] lambda;    // per-capita intrinsic rate of increase
  vector<lower=0>[nsp] es;        // competitive EFFECTS of each species
  vector<lower=0>[nsp] rs;        // competitive RESPONSES of each species
  real<lower=0> sigma;            // sd of alpha_ij deviations from r_i*e_j
}

transformed parameters {
  vector<lower=0>[nalphas] alphadevs; // alpha deviations from rs*es
  matrix<lower=0>[nsp,nsp] devmat = to_matrix(alphadevs, nsp, nsp);
  matrix[nsp,nsp] alpha = rs * es' + devmat;  // interaction matrix
  vector<lower=0>[N] fec;                       // expected fecundities
  for(i in 1:N) {
    fec[i] = lambda[focalsp[i]] / (1. + bkgd_dens[i] * alpha[focalsp[i]]');
  }
}

model {
  offspring ~ poisson(fec);      // observe expected fecundity
  lambda ~ normal(lambda_mean, lambda_sd);  // normal prior on lambda
  es ~ normal(es_mean, es_sd);   // normal prior on competitive EFFECTS
  rs ~ normal(rs_mean, rs_sd);   // normal prior on competitive RESPONSES
  alphadevs ~ normal(0, sigma);
  sigma ~ cauchy(0, alphasd_scale);    // [hyper prior] 
}
