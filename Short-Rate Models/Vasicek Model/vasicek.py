import numpy as np
import pandas as pd



class Vasicek:
    """
    Represents an instance of the vasicek model
    """
    def __init__(self, kappa = 1, theta = 3, sigma = 0.5):
        """
        Creates an instance of the vasicek model and initialises its parameters.
        Arguments:
            kappa : the speed of mean reversion.
            theta : the long term rate.
            sigma : the volatility parameter.
        """
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        
        
    def euler_simulation(self, rt, n_years, n_timesteps, n_scenarios):
        """
        Simulates sample paths from the vasicek model using an euler discretisation scheme.
        Arguments:
            rt : the instantaneous rate at time t.
            n_years : the number of years from time t.
            n_timesteps : the number of time steps in T.
            n_scenarios : the number of scenarios to generate.
        Returns:
            A dataframe of sample paths of the instantaneous spot rate
        """
        dt = n_years / n_timesteps
        random_draws = np.random.normal(0, scale = np.sqrt(dt), size = (n_timesteps, n_scenarios))
        rates = np.empty_like(random_draws)
        rates[0] = rt

        for step in range(1, n_timesteps):
            rt = rates[step - 1]
            d_rt = self.kappa * (self.theta - rt) * dt + self.sigma * random_draws[step]
            rates[step] = rt + d_rt

        return pd.DataFrame(data = rates, index = range(n_timesteps))
    
    
    def moments_simulation(self, rt, n_years, n_timesteps, n_scenarios):
        """
        Simulates sample paths from the vasicek model using the moments of the conditional distribution of the spot rate
        Arguments:
            rt : the instantaneous rate at time t.
            n_years : the number of years from time t.
            n_timesteps : the number of time steps in T.
            n_scenarios : the number of scenarios to generate.
        Returns:
            A dataframe of sample paths of the instantaneous spot rate
        """
        dt = n_years / n_timesteps
        random_draws = np.random.normal(size = (n_timesteps, n_scenarios))
        sdev = self.sigma*np.sqrt((1 - np.exp(-2*self.kappa*dt))/(2*self.kappa))
        
        rates = np.empty_like(random_draws)
        rates[0] = rt

        for step in range(1, n_timesteps):
            rt = rates[step - 1]
            rates[step] = self.theta + (rates[step - 1] - self.theta)*np.exp(-self.kappa * dt) + sdev*random_draws[step]

        return pd.DataFrame(data = rates, index = range(n_timesteps))