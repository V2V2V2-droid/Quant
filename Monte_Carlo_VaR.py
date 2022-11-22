import numpy as np
import matplotlib.pyplot as plt
# MT VaR relies on normality assumption for return simulation:
# once the simulated M random walks are made, you can get the average returns and
# apply a simple percentile for the VaR x

# you need a m param for period of simulation: take m = 252 : number of banking days per year or m = 60
# T represent the number of days you will run the simulation for.



class MonteCarloVar:
    def __init__(self, mu: float = 0.05, sigma: float = 0.02, m: int = 60, S0: float = 220, n: int = 100, T:int =1):
        """
        Run a monte Carlo Var based on the given parameters
        Var output than has to be multiplied by portfolio USD value
        returns are portfolio returns
        :param mu: parametric mean (from historical returns)
        :param sigma: parametric (from historical returns)
        :param T: number of days we base the VAR calc on
        :param S0: initial price level / portfolio value (multivariate for multiple positions)
        :param n: number of simulations run
        """
        self.mu = mu
        self.sigma = sigma
        self.S0 = S0
        self.n = n
        self.m = m
        self.T = T
        self.random_returns = [np.random.normal(mu, sigma, self.m) for _ in range(n)]
        self.cumulative_returns = [(i + 1).cumprod() for i in self.random_returns]
        self.random_walks = [self.S0 * (j + 1).cumprod() for j in self.random_returns]


    def VaR(self, x: int = 5):
        self.avg_ret = np.mean(self.random_returns, axis=1)
        VaR = np.percentile(self.avg_ret, x)*np.sqrt(self.T)
        print("VaR for {}% is: {}".format(str(x), str(VaR)))
        return VaR

    def plot_random_walks(self):
        for i in range(self.n):
            plt.plot(range(self.m), self.random_walks[i])
            plt.title("Random walk with mean {} and std: {}".format(self.mu, self.sigma))
        plt.show()

    def plot_returns_trajectory(self):
        for k in range(self.n):
            plt.plot(range(self.m), self.cumulative_returns[k])
            plt.title("Cumulative return trajectory with mean {} and std: {}".format(self.mu, self.sigma))
        plt.show()


if __name__ == "__main__":
    a = MonteCarloVar(mu=0.09, sigma=0.05)
    print(a.VaR(x=5))
    a.plot_returns_trajectory()