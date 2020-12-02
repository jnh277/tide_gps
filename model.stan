
data {
    int<lower=0> N;
    vector[N] x;
    vector[N] y;
    real lambda1;
}

parameters {
    // set the upper limit according to nyquist sampling criteria?
    real<lower=50,upper=400> alpha;     // frequency constant
    real<lower=-200,upper=200> beta;      // frequency linear
    real mu;                            // signal mean
    real tau;                  // decay rate
    real A;                    // sin amplitude
    real B;                    // cos amplitude
    // does constraining these to postive make sense??
    real<lower=0.00001, upper=10.0> sig_e;          // noise variance
    real<lower=1.0,upper=100.0> nu;
    real<lower=0, upper=100.0> sig_lin;     //
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
//    y ~ normal(f, sig_e);
    y ~ student_t(nu, f, sig_e + sig_lin * x);


}
generated quantities {
    vector[N] h;
    vector[N] frequency;
    frequency = alpha + beta * x;
    h = lambda1 * frequency / (2 * pi()) /2;
}