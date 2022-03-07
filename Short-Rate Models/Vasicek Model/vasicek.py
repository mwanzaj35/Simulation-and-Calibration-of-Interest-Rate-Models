import numpy as np
import pandas as pd



class Vasicek:
    """
    Represents an instance of the vasicek model
    """
    def __init__(self, kappa = 1, theta = 3, sigma = 0.5):
        """
        kappa : the speed of mean reversion.
        theta : the long term rate.
        sigma : the volatility parameter.
        """
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        
    def simulate(self, rt, n_years, n_timesteps, n_scenarios):
        """
        rt : the instantaneous rate at time t.
        n_years : the number of years from time t.
        n_timesteps : the number of time steps in T.
        n_scenarios : the number of scenarios to generate.
        """
        dt = n_years / n_timesteps
        random_draws = np.random.normal(0, scale = np.sqrt(dt), size = (n_timesteps, n_scenarios))
        rates = np.empty_like(random_draws)
        rates[0] = rt

        for step in range(1, n_timesteps):
            rt = rates[step - 1]
            d_rt = self.kappa*(self.theta -rt)*dt + self.sigma*random_draws[step]
            rates[step] = rt + d_rt

        return pd.DataFrame(data = rates, index = range(n_timesteps))