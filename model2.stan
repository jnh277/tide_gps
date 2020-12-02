
data {
    int<lower=0> N;
    vector[N] x;
    vector[N] y;
    real lambda1;
}

parameters {
    // set the upper limit according to nyquist sampling criteria?
//    real<lower=50,upper=150> alpha;     // frequency constant
//    real<lower=0.1,upper=150> beta;      // frequency linear
    h0<lower=0.01,upper=10.0>               // height constant
    h1<lower=-20,upper=-20>
    real<lower=0.1,upper=4>
    real mu;                            // signal mean
    real<lower=0> tau;                  // decay rate
    real A;                    // sin amplitude
    real B;                    // cos amplitude
    // does constraining these to postive make sense??
    real<lower=0.00001> sig_e;          // noise variance
}
transformed parameters {
    vector[N] f;
    f = mu + exp(-tau * x) .* (A * sin(alpha * x + beta * x .* x ) + B * cos(alpha * x + beta * x .* x));
}

model {

    // priors
    alpha ~ cauchy(0.0, 1.0);
    beta ~ cauchy(0.0, 1.0);
    mu ~ cauchy(0.0, 1.0);
    A ~ cauchy(0.0, 1.0);
    B ~ cauchy(0.0, 1.0);
    sig_e ~ cauchy(0.0, 1.0);

    // likelihood model
    y ~ normal(f, sig_e);

}
//generated quantities {
//    vector[no_obs_val] fhat;
//    for (n in 1:no_obs_val) {
//        y_hat[n] = val_input_matrix[n, :] * b_coefs-val_obs_matrix[n, :]*a_coefs;
//    }
//
//}