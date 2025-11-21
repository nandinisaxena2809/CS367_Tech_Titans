"""
Financial Time Series Analysis using Gaussian Hidden Markov Models
Lab Assignment 5 - AI Course
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn import hmm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class FinancialHMM:
    def __init__(self, ticker, start_date, end_date, n_states=2):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.n_states = n_states
        self.df = None
        self.model = None
        
    def download_data(self):
        print(f"Downloading data for {self.ticker}...")
        self.df = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        print(f"Downloaded {len(self.df)} data points.")
        return self.df
    
    def preprocess_data(self):
        print("Preprocessing data...")
        self.df['adj_close'] = self.df['Close']
        self.df['log_return'] = np.log(self.df['adj_close'] / self.df['adj_close'].shift(1))
        self.df['simple_return'] = self.df['adj_close'].pct_change()
        self.df['volatility'] = self.df['log_return'].rolling(window=20).std()
        self.df = self.df.dropna()
        print(f"Data preprocessed. Shape: {self.df.shape}")
        print("\nData Summary:")
        print(self.df[['adj_close', 'log_return', 'volatility']].describe())
        return self.df
    
    def fit_hmm(self):
        print(f"\nFitting Gaussian HMM with {self.n_states} states...")
        X = self.df['log_return'].values.reshape(-1, 1)
        
        # fit the hmm model
        self.model = hmm.GaussianHMM(n_components=self.n_states, covariance_type="full",
                                     n_iter=1000, random_state=42, verbose=False)
        self.model.fit(X)
        
        # get hidden states
        self.df['state'] = self.model.predict(X)
        posteriors = self.model.predict_proba(X)
        for i in range(self.n_states):
            self.df[f'state_{i}_prob'] = posteriors[:, i]
        
        print("Model fitted successfully!")
        return self.model
    
    def analyze_parameters(self):
        print("\n" + "="*60)
        print("HMM PARAMETER ANALYSIS")
        print("="*60)
        
        means = self.model.means_.flatten()
        stds = np.sqrt(np.array([np.diag(cov) for cov in self.model.covars_])).flatten()
        state_order = np.argsort(means)
        
        print("\nHidden State Characteristics:")
        print("-" * 60)
        for idx, state in enumerate(state_order):
            mean_return = means[state] * 252 * 100  # annualized %
            volatility = stds[state] * np.sqrt(252) * 100
            
            # figure out what kind of market regime this is
            if mean_return > 0 and volatility < stds.mean() * np.sqrt(252) * 100:
                regime = "Bull Market (High return, Low volatility)"
            elif mean_return > 0:
                regime = "Volatile Bull Market"
            elif volatility < stds.mean() * np.sqrt(252) * 100:
                regime = "Bear Market (Low volatility)"
            else:
                regime = "Bear Market (High volatility)"
            
            print(f"\nState {state}:")
            print(f"  Daily Mean Return:      {means[state]:.6f}")
            print(f"  Daily Std Dev:          {stds[state]:.6f}")
            print(f"  Annualized Return:      {mean_return:.2f}%")
            print(f"  Annualized Volatility:  {volatility:.2f}%")
            print(f"  Interpretation:         {regime}")
            
            pct_time = (self.df['state'] == state).sum() / len(self.df) * 100
            print(f"  Time in State:          {pct_time:.2f}%")
        
        # transition probabilities
        print("\n" + "-" * 60)
        print("Transition Matrix:")
        print("-" * 60)
        print(pd.DataFrame(self.model.transmat_,
                          columns=[f"To State {i}" for i in range(self.n_states)],
                          index=[f"From State {i}" for i in range(self.n_states)]))
        
        # stationary dist
        print("\nStationary Distribution:")
        eigenvals, eigenvecs = np.linalg.eig(self.model.transmat_.T)
        stationary = eigenvecs[:, np.argmax(eigenvals)].real
        stationary = stationary / stationary.sum()
        for i, prob in enumerate(stationary):
            print(f"  State {i}: {prob:.4f} ({prob*100:.2f}%)")
        
        return means, stds
    
    def predict_future_state(self, n_days=5):
        print("\n" + "="*60)
        print("FUTURE STATE PREDICTION")
        print("="*60)
        
        current_state = self.df['state'].iloc[-1]
        print(f"\nCurrent State: {current_state}")
        
        # predict next states using transition matrix
        trans_matrix = self.model.transmat_
        current_dist = np.zeros(self.n_states)
        current_dist[current_state] = 1.0
        
        print(f"\nPredicted state probabilities for next {n_days} days:")
        for day in range(1, n_days + 1):
            current_dist = current_dist @ trans_matrix
            print(f"\nDay {day}:")
            for state in range(self.n_states):
                print(f"  State {state}: {current_dist[state]:.4f} ({current_dist[state]*100:.2f}%)")
    
    def visualize_results(self):
        fig = plt.figure(figsize=(16, 12))
        palette = sns.color_palette("tab10", self.n_states)
        
        # price chart with states
        ax1 = plt.subplot(4, 1, 1)
        for state in range(self.n_states):
            mask = self.df['state'] == state
            ax1.scatter(self.df.index[mask], self.df['adj_close'][mask],
                       c=[palette[state]], label=f'State {state}', alpha=0.6, s=10)
        ax1.set_title(f'{self.ticker} Price with Hidden Market Regimes', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # returns
        ax2 = plt.subplot(4, 1, 2)
        for state in range(self.n_states):
            mask = self.df['state'] == state
            ax2.scatter(self.df.index[mask], self.df['log_return'][mask],
                       c=[palette[state]], label=f'State {state}', alpha=0.6, s=10)
        ax2.set_title('Daily Log Returns with Hidden States', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Log Return', fontsize=12)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # state probs over time
        ax3 = plt.subplot(4, 1, 3)
        for state in range(self.n_states):
            ax3.plot(self.df.index, self.df[f'state_{state}_prob'],
                    label=f'State {state}', alpha=0.7)
        ax3.set_title('State Probabilities Over Time', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Probability', fontsize=12)
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 1])
        
        # volatilty
        ax4 = plt.subplot(4, 1, 4)
        for state in range(self.n_states):
            mask = self.df['state'] == state
            ax4.scatter(self.df.index[mask], self.df['volatility'][mask],
                       c=[palette[state]], label=f'State {state}', alpha=0.6, s=10)
        ax4.set_title('20-Day Rolling Volatility with Hidden States', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Date', fontsize=12)
        ax4.set_ylabel('Volatility', fontsize=12)
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.ticker}_hmm_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved as '{self.ticker}_hmm_analysis.png'")
        plt.show()
        
        self._plot_state_distribution()
    
    def _plot_state_distribution(self):
        fig, axes = plt.subplots(1, self.n_states, figsize=(14, 4))
        if self.n_states == 1:
            axes = [axes]
        
        # plot histogram for each state
        for state in range(self.n_states):
            mask = self.df['state'] == state
            returns = self.df.loc[mask, 'log_return']
            axes[state].hist(returns, bins=50, alpha=0.7, color=sns.color_palette("tab10")[state])
            axes[state].axvline(returns.mean(), color='red', linestyle='--', 
                              label=f'Mean: {returns.mean():.6f}')
            axes[state].set_title(f'State {state} Return Distribution')
            axes[state].set_xlabel('Log Return')
            axes[state].set_ylabel('Frequency')
            axes[state].legend()
            axes[state].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.ticker}_state_distributions.png', dpi=300, bbox_inches='tight')
        print(f"State distribution saved as '{self.ticker}_state_distributions.png'")
        plt.show()
    
    def evaluate_model(self):
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        X = self.df['log_return'].values.reshape(-1, 1)
        log_likelihood = self.model.score(X)
        print(f"\nLog-Likelihood: {log_likelihood:.2f}")
        
        # calc aic and bic
        n_params = self.n_states * self.n_states - self.n_states + self.n_states * 2
        aic = 2 * n_params - 2 * log_likelihood
        bic = n_params * np.log(len(X)) - 2 * log_likelihood
        print(f"AIC: {aic:.2f}")
        print(f"BIC: {bic:.2f}")
        
        # how long do states last on avg
        state_changes = (self.df['state'].diff() != 0).sum()
        avg_persistence = len(self.df) / (state_changes + 1)
        print(f"\nAverage State Persistence: {avg_persistence:.2f} days")
        
        return log_likelihood, aic, bic
    
    def run_full_analysis(self):
        print("\n" + "="*70)
        print("FINANCIAL TIME SERIES ANALYSIS USING GAUSSIAN HIDDEN MARKOV MODELS")
        print("="*70)
        
        self.download_data()
        self.preprocess_data()
        self.fit_hmm()
        self.analyze_parameters()
        self.evaluate_model()
        self.predict_future_state()
        self.visualize_results()
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)


def compare_multiple_states(ticker, start_date, end_date, state_range=[2, 3, 4]):
    print("\n" + "="*70)
    print("COMPARING MODELS WITH DIFFERENT NUMBERS OF STATES")
    print("="*70)
    
    results = []
    for n_states in state_range:
        print(f"\n\nTesting with {n_states} states...")
        analyzer = FinancialHMM(ticker, start_date, end_date, n_states=n_states)
        analyzer.download_data()
        analyzer.preprocess_data()
        analyzer.fit_hmm()
        
        X = analyzer.df['log_return'].values.reshape(-1, 1)
        log_likelihood = analyzer.model.score(X)
        n_params = n_states * n_states - n_states + n_states * 2
        aic = 2 * n_params - 2 * log_likelihood
        bic = n_params * np.log(len(X)) - 2 * log_likelihood
        
        results.append({'n_states': n_states, 'log_likelihood': log_likelihood,
                       'AIC': aic, 'BIC': bic})
    
    comparison_df = pd.DataFrame(results)
    print("\n\nModel Comparison:")
    print(comparison_df.to_string(index=False))
    
    # find best
    best_aic = comparison_df.loc[comparison_df['AIC'].idxmin()]
    best_bic = comparison_df.loc[comparison_df['BIC'].idxmin()]
    print(f"\nBest model by AIC: {int(best_aic['n_states'])} states")
    print(f"Best model by BIC: {int(best_bic['n_states'])} states")
    
    return comparison_df


def main():
    # config
    TICKER = "AAPL"  # change to ^GSPC for S&P 500, TSLA for Tesla, etc
    START_DATE = "2013-01-01"
    END_DATE = datetime.today().strftime("%Y-%m-%d")
    N_STATES = 2
    
    # run analysis
    analyzer = FinancialHMM(TICKER, START_DATE, END_DATE, n_states=N_STATES)
    analyzer.run_full_analysis()
    
    # compare different state counts
    print("\n\n" + "="*70)
    print("BONUS: COMPARING DIFFERENT MODEL COMPLEXITIES")
    print("="*70)
    comparison = compare_multiple_states(TICKER, START_DATE, END_DATE, state_range=[2, 3, 4])
    
    # save
    analyzer.df.to_csv(f'{TICKER}_hmm_results.csv')
    print(f"\n\nResults saved to '{TICKER}_hmm_results.csv'")


if __name__ == "__main__":
    main()
