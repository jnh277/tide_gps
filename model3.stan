
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
    real c0;                            // signal mean
    real c1;                            // signal linear coefficient
    real c2;                            // signal quadratic coefficient
    real<lower=0> tau;                  // decay rate
    real A;                    // sin amplitude
    real B;                    // cos amplitude
    // does constraining these to postive make sense??
    real<lower=0.00001, upper=10.0> sig_e;          // noise variance
    real<lower=1.0,upper=100.0> nu;
 //   real<lower=0, upper=100.0> sig_lin;     //
}
transformed parameters {
    vector[N] f;
    f = c0 + c1 * x + c2 * (x .* x) + exp(-tau * x) .* (A * sin(alpha * x + beta * x .* x ) + B * cos(alpha * x + beta * x .* x));
}

model {

    // priors
    alpha ~ cauchy(100., 30.0);      // 100 roughly corresponds to a mean height of 1.5m
    beta ~ cauchy(0.0, 60.0);
    c0 ~ cauchy(0.0, 0.1);
    c1 ~ cauchy(0.0, 1.0);
    c2 ~ cauchy(0.0, 3.0);
    A ~ cauchy(0.0, 2.0);
    B ~ cauchy(0.0, 2.0);
    sig_e ~ cauchy(0.0, 1.0);
    tau ~ cauchy(0.0, 1.0);

    // likelihood model
//    y ~ normal(f, sig_e);
    y ~ student_t(nu, f, sig_e);


}
generated quantities {
    vector[N] h;
    vector[N] frequency;
    frequency = alpha + beta * x;
    h = lambda1 * frequency / (2 * pi()) /2;
}