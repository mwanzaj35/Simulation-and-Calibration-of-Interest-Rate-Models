import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression


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
        
        rates = np.empty([n_timesteps + 1, n_scenarios])
        rates[0] = rt

        for step in range(1, n_timesteps + 1):
            rt = rates[step - 1]
            d_rt = self.kappa * (self.theta - rt) * dt + self.sigma * random_draws[step - 1]
            rates[step] = rt + d_rt

        return pd.DataFrame(data = rates, index = range(n_timesteps + 1))
    
    
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
        
        rates = np.empty([n_timesteps + 1, n_scenarios])
        rates[0] = rt

        for step in range(1, n_timesteps + 1):
            rt = rates[step - 1]
            rates[step] = self.theta + (rates[step - 1] - self.theta)*np.exp(-self.kappa * dt) + sdev*random_draws[step - 1]

        return pd.DataFrame(data = rates, index = range(n_timesteps + 1))
    
    
    def regression_calibration(self, data, dt):
        """
        Calibrates the parameters of the vasicek model using a linear regression
        Arguments:
            data : a time series of instantaneous spot rates at time t.
            dt: time difference between consecutive data points as a year fraction
        Returns:
            changes the parameters of the vasicek model
        """
        y = data[1:]
        X = np.array(data[:-1]).reshape(-1, 1)
        
        reg = LinearRegression().fit(X, y)
        a = reg.coef_[0]
        b = reg.intercept_
        std_resid = np.sqrt(sum((y - reg.predict(X))**2)/(len(y) - 2))
        
        self.kappa = -np.log(a)/dt
        self.theta = b / (1 - a)
        self.sigma = std_resid*np.sqrt((-2*np.log(a))/(dt*(1-a**2)))
        
        
    def mle_calibration(self, data, dt):
        """
        Calibrates the parameters of the vasicek model using maximum likelihood estimation
        Arguments:
            data : a time series of instantaneous spot rates at time t.
            dt: time difference between consecutive data points as a year fraction
        Returns:
            changes the parameters of the vasicek model
        """
        n = len(data) - 1    
        Sx = np.sum(data[:-1])
        Sy = np.sum(data[1:])
        Sxx = np.sum(data[:-1]**2)
        Syy = np.sum(data[1:]**2)
        Sxy = np.sum(np.multiply(data[:-1], data[1:]))
        
        theta = (Sy*Sxx - Sx*Sxy) / (n*(Sxx - Sxy) - Sx**2 + Sx*Sy)
        
        a = (Sxy - theta*Sx - theta*Sy + n*theta**2)/(Sxx - 2*theta*Sx + n*theta**2)
        kappa = -np.log(a)/dt
        
        sigma_bar2 = (Syy - 2*a*Sxy + a**2*Sxx - 2*theta*(1 - a)*(Sy - a*Sx) + n*theta**2*(1 - a)**2)/(n - 1)
        sigma = np.sqrt(sigma_bar2*2*kappa/(1 - a**2))
        
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
    
    
    def __BtT(self, t, T):
        
        return (1 / self.kappa) * (1 - np.exp(- self.kappa * (T - t)))
    
    
    def __AtT(self, t, T):
        
        at1 = self.theta - 0.5*(self.sigma / self.kappa)**2
        at2 = self.__BtT(t, T) - T + t
        at3 = (self.sigma**2 / 4*self.kappa) * self.__BtT(t, T)**2
        
        return np.exp(at1 * at2 - at3)
    
    
    def analytical_bondprice(self, rt, t, T):
        # to do: add numerical integration to solve for bond price
        return self.__AtT(t, T) * np.exp(- rt * self.__BtT(t, T))
    
    
    def analytical_optionprice(self, rt, t, T, S, K):
        # to do: add numerical integration to solve for bond option price
        sigma_p = self.sigma * np.sqrt((1 - np.exp(-2*self.kappa*(T - t)))/(2*self.kappa)) * self.__BtT(T, S)
        d1 = (np.log(self.analytical_bondprice(rt, t, S) / (self.analytical_bondprice(rt, t, T) * K)) * (0.5 * sigma_p**2)) / sigma_p
        d2 = d1 - sigma_p
        
        return self.analytical_bondprice(rt, t, S)* stats.norm.cdf(d1) - K * self.analytical_bondprice(rt, t, T)* stats.norm.cdf(d2)
    
    
    
        
        
                                       
       
                                       
                                      
                                                   
                                                   
                                                   
                                                   