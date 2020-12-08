
data {
    int<lower=0> N;
    vector[N] x;
    vector[N] y;
    vector[N] t;
    real lambda1;
}

parameters {
    // set the upper limit according to nyquist sampling criteria?

    real<lower=0,upper=5> h;                           // current height
//    real<lower=0.001> sig_h;
//    vector[N] dh;
    // Principle diurnal harmonic M1
//    real<lower=2*pi()/14,upper=2*pi()/10> M1_freq;     // frequency between 11 and 12 hour period
//    real<lower=-mu,upper=mu> M1_mag;                   // sin amplitude
//    real<lower=-pi(),upper=pi()> M1_phase;              // cos amplitude

    // signal parameters
    real mu;                    // signal mean
    real A;                                         // sin amplitude
    real B;                                         // cos amplitude
    real<lower=0> tau;                              // decay rate
    real<lower=0.00001, upper=10.0> sig_e;          // noise variance
    real<lower=1.0,upper=100.0> nu;                 // student's T degrees of freedom
//    real<lower=0, upper=100.0> sig_lin;             // linear dependent noise, don't include for now
}
transformed parameters {
//    vector [N] h = M1_mag * sin(M1_freq * t + M1_phase); //+ B * cos(M1_freq * t);
    real sf = h / lambda1 * 4 * pi();     // signal frequency
    vector[N] signal;
    signal = mu + exp(-tau * x) .* (A * sin(sf * x ) + B * cos(sf * x));
}

model {

    // priors

    mu ~ cauchy(0.0, 1.0);
    A ~ cauchy(0.0, 1.0);
    B ~ cauchy(0.0, 1.0);
    sig_e ~ cauchy(0.0, 1.0);
//    sig_h ~ cauchy(0.0, 1.0);

    // process model
//    h[2:N] ~ normal(h[1:N-1], sig_h);

    // likelihood model
//    y ~ normal(f, sig_e);
    y ~ student_t(nu, signal, sig_e);


}
//generated quantities {
//    vector[N] h;
//    vector[N] frequency;
//    frequency = alpha + beta * x;
//    h = lambda1 * frequency / (2 * pi()) /2;
//}