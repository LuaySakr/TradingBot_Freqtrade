import math
from datetime import datetime
 
from pandas import DataFrame, date_range
from statistics import pstdev
 
from freqtrade.optimize.hyperopt import IHyperOptLoss
 
resample_freq = '1D'
slippage_per_trade_ratio = 0.0005
days_in_year = 365
sqrt365 = math.sqrt(days_in_year)
MAX_TRADE_DURATION = 300    # minutes (5m * 24smp * 3shift * 0.75wr)
mar = 0.0
 
 
class SortinoLossDaily(IHyperOptLoss):
    """
    Defines the loss function for hyperopt.
    This implementation uses the Sortino Ratio calculation.
    """
 
    def hyperopt_loss_function(results: DataFrame, trade_count: int, min_date: datetime,
                               max_date: datetime, *args, **kwargs) -> float:
        trade_duration = results.trade_duration.values.mean()
        if trade_duration > MAX_TRADE_DURATION:
            return 20
        # apply slippage per trade to profit_percent
        results.loc[:, 'profit_percent_after_slippage'] = \
            results['profit_percent'] - slippage_per_trade_ratio
 
        # create the index within the min_date and end max_date
        t_index = date_range(start=min_date, end=max_date, freq=resample_freq, normalize=True)
 
        sum_daily = (results.resample(resample_freq, on='close_time').agg({
            "profit_percent_after_slippage":
            sum
        }).reindex(t_index).fillna(0))
 
        total_profit = sum_daily["profit_percent_after_slippage"]
        expected_returns_mean = total_profit.mean() - mar
 
        sum_daily['downside_returns'] = mar
        sum_daily.loc[total_profit < mar,
                      'downside_returns'] = sum_daily['profit_percent_after_slippage']
        total_downside = sum_daily['downside_returns']
 
        down_stdev = pstdev(total_downside, mar)
 
        if (down_stdev != 0.):
            sortino_ratio = expected_returns_mean / down_stdev * sqrt365
        else:
            # Define high (negative) sortino ratio to be clear that this is NOT optimal.
            sortino_ratio = -20.
 
        # print(t_index, sum_daily, total_profit)
        # print(risk_free_rate, expected_returns_mean, down_stdev, sortino_ratio)
        return -sortino_ratio