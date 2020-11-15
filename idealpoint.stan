//
// Ideal Point Multilevel Modeling and Postratification
//


data {
  int<lower=1> J; //Participants
  int<lower=1> K; //Questions
  int<lower=1> N; //no. of observations
  int<lower=1> S; //no. of states
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

  //real mu_alpha;
  //real<lower=0> sigma_alpha;
  real mu_beta;
  real<lower=0> sigma_beta;

  vector[K] beta_raw;
  vector[J] alpha_raw;
  vector<lower=0>[K] theta;
}
transformed parameters{
  vector[6] alpha_age = 0 + sigma_age*alpha_age_raw;
  vector[5] alpha_educ = 0 + sigma_educ*alpha_educ_raw;
  vector[4] alpha_ethnicity = 0 + sigma_ethnicity*alpha_ethnicity_raw;
  vector[4] alpha_region = 0 + sigma_region*alpha_region_raw;
  vector[K] beta = mu_beta + sigma_beta*beta_raw;

  vector[S] alpha_state;
  vector[J] alpha;

  real mu_alpha = 0; // constraint on mu_alpha
  real sigma_alpha = 1; // constraint on sigma_alpha

  for(s in 1:S)
    alpha_state[s] = alpha_region[region[s]] + beta_repvote*repvote[s] + sigma_state*alpha_state_raw[s];
  for (j in 1:J)
    alpha[j] = mu_alpha + alpha_state[state[j]] + alpha_age[age[j]] + alpha_ethnicity[ethnicity[j]] + alpha_educ[educ[j]] + beta_male*male[j] + sigma_alpha*alpha_raw[j];
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
  sigma_beta ~ exponential(0.5); // prior for sigma_beta
  theta ~ normal(0, 1); // prior for theta

  alpha_state_raw ~ std_normal(); // implies alpha_state ~ normal(alpha_region, sigma_state)
  alpha_age_raw ~ std_normal(); // implies alpha_age ~ normal(0, sigma_age)
  alpha_ethnicity_raw ~ std_normal(); // implies alpha_ethnicity ~ normal(0, sigma_ethnicity)
  alpha_educ_raw ~ std_normal(); // implies alpha_educ ~ normal(0, sigma_educ)
  alpha_region_raw ~ std_normal(); // implies alpha_region ~ normal(0, sigma_region)

  beta_raw ~ std_normal(); // implies beta ~ normal(mu_beta, sigma_beta)
  alpha_raw ~ std_normal(); // implies alpha ~ normal(mu_alpha + alpha_state + alpha_age + ..., sigma_alpha)
  for (n in 1:N)
    y[n] ~ bernoulli_logit(theta[question[n]] * (alpha[participant[n]] - beta[question[n]]));
}

generated quantities{
  vector[S*6*5*4*2] alpha_pred;
  vector[S*6*5*4*2] abortion1_pred;
  vector[S*6*5*4*2] abortion2_pred;
  vector[S*6*5*4*2] abortion3_pred;
  vector[S*6*5*4*2] abortion4_pred;
  vector[S*6*5*4*2] abortion5_pred;

  for (st in 1:S)
    for (ag in 1:6)
      for (ed in 1:5)
        for (et in 1:4)
          for (ge in 1:2)
            alpha_pred[1 + (st-1)*12000/50 + (ag-1)*12000/50/6 + (ed-1)*12000/50/6/5 + (et-1)*12000/50/6/5/4 + (ge-1)*12000/50/6/5/4/2] = mu_alpha +
              alpha_state[st] + alpha_age[ag] + alpha_ethnicity[et] + alpha_educ[ed] + beta_male*(ge-1.5);

  abortion1_pred = inv_logit(theta[1]*(alpha_pred - beta[1]));
  abortion2_pred = inv_logit(theta[2]*(alpha_pred - beta[2]));
  abortion3_pred = inv_logit(theta[3]*(alpha_pred - beta[3]));
  abortion4_pred = inv_logit(theta[4]*(alpha_pred - beta[4]));
  abortion5_pred = inv_logit(theta[5]*(alpha_pred - beta[5]));
}
