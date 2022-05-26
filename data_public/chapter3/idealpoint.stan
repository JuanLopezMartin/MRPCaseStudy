//
// Ideal Point Multilevel Modeling and Postratification
//


data {
  int<lower=1> J; //Participants
  int<lower=1> K; //Questions
  int<lower=1> N; //no. of observations
  int<lower=1> S; //no. of states
  int<lower=1> P; //no. of states
  int<lower=1, upper=J> participant[N]; // Participant for observation n
  int<lower=1, upper=K> question[N]; // Question for observation n
  int<lower=1, upper=S> state[N]; // State for observation n
  int<lower=1, upper=6> age[N]; // Age for observation n
  int<lower=1, upper=4> ethnicity[N]; // Ethnicity for observation n
  int<lower=1, upper=5> educ[N]; // Education for observation n
  real<lower=-0.5, upper=0.5> male[N]; // Gender for observation n
  int<lower=0, upper=4> region[S]; // Region for state s
  real repvote[S]; // Republican voteshare for state s
  int<lower=0, upper=1> y[N]; // Support for observation n
  int<lower=1, upper=S> postrat_state[P];
  int<lower=1, upper=6> postrat_age[P];
  int<lower=1, upper=4> postrat_ethnicity[P];
  int<lower=1, upper=5> postrat_educ[P];
  real<lower=-0.5, upper=0.5> postrat_male[P];
}
parameters {
  vector[S] alpha_state_raw;
  vector[6] alpha_age_raw;
  vector[5] alpha_educ_raw;
  vector[4] alpha_ethnicity_raw;
  vector[4] alpha_region_raw;
  real beta_male;
  real beta_repvote;
  real<lower=0> sigma_state;
  real<lower=0> sigma_age;
  real<lower=0> sigma_ethnicity;
  real<lower=0> sigma_educ;
  real<lower=0> sigma_region;

  real mu_alpha;
  real<lower=0> sigma_alpha;
  real mu_beta;
  real<lower=0> sigma_beta;
  real<lower=0> mu_gamma;
  real<lower=0> sigma_gamma;

  vector[K] beta_raw;
  vector[J] alpha_raw;
  vector<lower=0>[K] gamma_raw;
}
transformed parameters{
  vector[6] alpha_age = 0 + sigma_age*alpha_age_raw;
  vector[5] alpha_educ = 0 + sigma_educ*alpha_educ_raw;
  vector[4] alpha_ethnicity = 0 + sigma_ethnicity*alpha_ethnicity_raw;
  vector[4] alpha_region = 0 + sigma_region*alpha_region_raw;
  vector[K] beta = mu_beta + sigma_beta*beta_raw;
  vector[K] gamma = mu_gamma + sigma_gamma*gamma_raw;

  vector[S] alpha_state;
  vector[J] alpha;

  real alpha_mean;
  real alpha_sd;
  vector[J] alpha_adj;
  vector[K] beta_adj;
  vector<lower=0>[K] gamma_adj;

  for(s in 1:S)
    alpha_state[s] = alpha_region[region[s]] + beta_repvote*repvote[s] + sigma_state*alpha_state_raw[s];
  for (j in 1:J)
    alpha[j] = mu_alpha + alpha_state[state[j]] + alpha_age[age[j]] + alpha_ethnicity[ethnicity[j]] + alpha_educ[educ[j]] + beta_male*male[j] + sigma_alpha*alpha_raw[j];

  alpha_mean = mean(alpha);
  alpha_sd = sd(alpha);
  alpha_adj = (alpha - alpha_mean)/alpha_sd;
  beta_adj = (beta - alpha_mean)/alpha_sd;
  gamma_adj = gamma*alpha_sd;
}



model {
  //priors on predictors
  sigma_state ~ exponential(0.5); // prior for sigma_state
  sigma_age ~ exponential(0.5); // prior for sigma_age
  sigma_ethnicity ~ exponential(0.5); // prior for sigma_ethnicity
  sigma_educ ~ exponential(0.5); // prior for sigma_educ
  sigma_region ~ exponential(0.5); // prior for sigma_educ
  beta_male ~ normal(0, 2); // prior for beta_male
  beta_repvote ~ normal(0, 2); // prior for beta_repvote

  //priors on parameters
  mu_beta ~ normal(0, 2); // prior for mu_beta
  sigma_beta ~ exponential(1); // prior for sigma_beta
  mu_gamma ~ normal(0, 2); // prior for mu_gamma
  sigma_gamma ~ exponential(1); // prior for sigma_gamma

  alpha_state_raw ~ std_normal(); // implies alpha_state ~ normal(alpha_region, sigma_state)
  alpha_age_raw ~ std_normal(); // implies alpha_age ~ normal(0, sigma_age)
  alpha_ethnicity_raw ~ std_normal(); // implies alpha_ethnicity ~ normal(0, sigma_ethnicity)
  alpha_educ_raw ~ std_normal(); // implies alpha_educ ~ normal(0, sigma_educ)
  alpha_region_raw ~ std_normal(); // implies alpha_region ~ normal(0, sigma_region)

  gamma_raw ~ std_normal(); // implies beta ~ normal(mu_beta, sigma_beta)
  beta_raw ~ std_normal(); // implies beta ~ normal(mu_beta, sigma_beta)
  alpha_raw ~ std_normal(); // implies alpha ~ normal(mu_alpha + alpha_state + alpha_age + ..., sigma_alpha)
  for (n in 1:N)
    y[n] ~ bernoulli_logit(gamma_adj[question[n]] * (alpha_adj[participant[n]] - beta_adj[question[n]]));
}

generated quantities{
  vector[P] alpha_pred_raw;
  vector[P] alpha_pred;


  for (p in 1:P)
    alpha_pred_raw[p] = alpha_state[postrat_state[p]] + alpha_age[postrat_age[p]] + alpha_ethnicity[postrat_ethnicity[p]] + alpha_educ[postrat_educ[p]] + beta_male*postrat_male[p];

  alpha_pred = (alpha_pred_raw - mean(alpha_pred_raw)) / sd(alpha_pred_raw);
}
