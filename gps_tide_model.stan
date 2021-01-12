
data {
    int<lower=0> sets;              // number of data sets
    int<lower=0> max_N;             // maximum set length
    int<lower=0> N[sets];           // length of each data set
    matrix[sets, max_N] x;
    matrix[sets, max_N] y;
    matrix[sets, max_N] t;
    real lambda1;
}

parameters {
    // tide model parameters
    real<lower=0, upper=6> h0;                          // mean tide height

    // Principle semi diurnal solar harmonic
    real<lower=2*pi()/15,upper=2*pi()/9> M2_freq;     // frequency between 10 and 14 hour period
    real M2_A;                   // sin amplitude
    real M2_B;              // cos amplitude

    // Principle semi diurnal lunar harmonic
    real<lower=2*pi()/15,upper=2*pi()/9> S2_freq;     // frequency between 10 and 14 hour period
    real S2_A;                   // sin amplitude
    real S2_B;              // cos amplitude

    // signal parameters
    vector[sets] mu;                                        // signal mean
    vector[sets] A;                                         // sin amplitude
    vector[sets] B;                                         // cos amplitude
    vector<lower=0>[sets] tau;                              // decay rate
    vector<lower=0.00001, upper=10.0>[sets] sig_e;          // noise variance
    vector<lower=1.0,upper=100.0>[sets] nu;                 // student's T degrees of freedom
}
transformed parameters {
    matrix[sets, max_N] h; //= h0 + M2_A * sin(M2_freq * t) + M2_B * cos(M2_freq * t) + S2_A * sin(M2_freq * t) + S2_B * cos(S2_freq * t);
    matrix[sets, max_N] sf; // = h / lambda1 * 4 * pi();     // signal frequency
    matrix[sets, max_N] signal;
//    signal = rep_matrix(mu,max_N) + exp(-rep_matrix(tau,max_N) .* x) .* (rep_matrix(A,max_N) .* sin(sf .* x ) + rep_matrix(B,max_N) .* cos(sf .* x));

    for (s in 1:sets)
    {
        h[s, 1:N[s]] = h0 + M2_A * sin(M2_freq * t[s, :N[s]]) + M2_B * cos(M2_freq * t[s, :N[s]]) + S2_A * sin(M2_freq * t[s,:N[s]])+ S2_B * cos(S2_freq * t[s, :N[s]]);
        sf[s, 1:N[s]] = h[s, 1:N[s]] / lambda1 * 4 *pi();
        signal[s, 1:N[s]] = mu[s] + exp(-tau[s] * x[s,:N[s]]) .* (A[s] * sin(sf[s,:N[s]] .* x[s,:N[s]] ) + B[s] * cos(sf[s,:N[s]] .* x[s,:N[s]]));
    }
}

model {
    // tide priors
    M2_A ~ cauchy(0, 5.0);
    M2_B ~ cauchy(0, 5.0);
    S2_A ~ cauchy(0, 5.0);
    S2_B ~ cauchy(0, 5.0);
    M2_freq ~ cauchy(2*pi()/12, 0.2);
    S2_freq ~ cauchy(2*pi()/12.5, 0.2);

    // signal priors
    mu ~ cauchy(0.0, 0.1);
    A ~ cauchy(0.0, 2.0);
    B ~ cauchy(0.0, 2.0);
    sig_e ~ cauchy(0.0, 1.0);
    tau ~ cauchy(0.0, 1.0);

    for (s in 1:sets){
        y[s,1:N[s]] ~ student_t(nu[s], signal[s,1:N[s]], sig_e[s]);
    }


}
