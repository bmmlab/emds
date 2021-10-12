data {
  int<lower=1> N;
  int<lower=1> T;
  int<lower=1> Tpart[N];
  int<lower=0, upper=1> choice[N, T]; // 
  matrix[T,4] TP[N]; // trial parameters
}

transformed data {
}

parameters {
// Declare all parameters as vectors for vectorizing
  // Hyper(group)-parameters
  vector[2] mu_pr;
  vector<lower=0>[2] sigma;

  // Subject-level raw parameters 
  vector[N] r_pr;
  vector[N] beta_pr;
}

transformed parameters {
  // Transform subject-level raw parameters
  vector<lower=0, upper=2>[N] r;
  vector<lower=0, upper=5>[N] beta;

  for (i in 1:N) {
    r[i]    = Phi_approx(mu_pr[1] + sigma[1] * r_pr[i]) * 2;
    beta[i] = Phi_approx(mu_pr[2] + sigma[2] * beta_pr[i]) * 5;
  }
}

model {
  // Hyperparameters
  mu_pr  ~ normal(0, 1);
  sigma  ~ normal(0, 0.2);

  // individual parameters
  r_pr    ~ normal(0, 1);
  beta_pr ~ normal(0, 1);

  for (i in 1:N) {
    // initialise variables to compute
    int tn;
    real u0;
    real u1;
    matrix[T,4] tp;
    int tmax;
    
    tp = TP[i];
    tmax = Tpart[i];
    
    for (t in 1:tmax) {
      tn = t;

      // compute utilities
      u0 = (1 / pow(1 + r[i] / 365, tp[tn, 2])) * tp[tn, 1];
      u1 = (1 / pow(1 + r[i] / 365, tp[tn, 4])) * tp[tn, 3];

      // compute decision probabilities
      choice[i, t] ~ bernoulli_logit( beta[i] * (u1 - u0) );
    }
  }
}

generated quantities {
  // For group level parameters
  real<lower=0, upper=2> mu_r;
  real<lower=0, upper=5> mu_beta;

  // For log likelihood calculation
  real log_lik[N];

  // For posterior predictive check
  real y_pred[N, T];

  // Set all posterior predictions to 0 (avoids NULL values)
  for (i in 1:N) {
    for (t in 1:T) {
      y_pred[i, t] = -1;
    }
  }

  mu_r    = Phi_approx(mu_pr[1]) * 2;
  mu_beta = Phi_approx(mu_pr[2]) * 5;

  { // local section, this saves time and space
    for (i in 1:N) {
      // initialise variables to compute
      int tn;
      real u0;
      real u1;
      matrix[T,4] tp;
      int tmax;
    
      tp = TP[i];
      tmax = Tpart[i];

      log_lik[i] = 0;

      for (t in 1:T) {
        tn = t;
        
        u0 = (1 / pow(1 + r[i] / 365, tp[tn, 2])) * tp[tn, 1];
        u1 = (1 / pow(1 + r[i] / 365, tp[tn, 4])) * tp[tn, 3];
        log_lik[i] += bernoulli_logit_lpmf(choice[i, t] | beta[i] * (u1 - u0));

        // generate posterior prediction for current trial
        y_pred[i, t] = bernoulli_rng(inv_logit(beta[i] * (u1 - u0)));
      }
    }
  }
}
