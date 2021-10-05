data {
    int<lower=0> T; // number of trials
    int<lower=0,upper=1> choices[T]; // choices
    real<lower=0> tp[T,4]; // trial parameters
}

parameters {
    real<lower=0> r;
    real<lower=0> beta;
}

model {
    // r ~ gamma(1, 2);
    // beta ~ gamma(0.5, 1);
    target += gamma_lpdf(r | 1, 2);
    target += gamma_lpdf(beta | 0.5, 1);
    
    for (t in 1:T) {
    
        real u0;
        real u1;
        
        u0 = tp[t, 2] * (1 / (1 + r / 12) ^ (tp[t, 1]));
        u1 = tp[t, 4] * (1 / (1 + r / 12) ^ (tp[t, 3]));
        
        // compute decision probabilities
        // choices[t] ~ bernoulli_logit( beta * (u1 - u0) ); 
        target += bernoulli_logit_lpmf(choices[t] | beta * (u1 - u0));

    }
}
