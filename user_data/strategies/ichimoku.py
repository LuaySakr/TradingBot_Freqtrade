import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from technical.indicators import ichimoku
from freqtrade.strategy.interface import IStrategy

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


# This class is a sample. Feel free to customize it.
class Ichimoku(IStrategy):
    """
    This is a sample strategy to inspire you.
    More information in https://www.freqtrade.io/en/latest/strategy-customization/

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the prototype for the methods: minimal_roi, stoploss, populate_indicators, populate_buy_trend,
    populate_sell_trend, hyperopt_space, buy_strategy_generator
    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 2

    minimal_roi = {
        "0": 1
    }
    stoploss = -0.1

    trailing_stop = True
    trailing_stop_positive = 0.15
    trailing_stop_positive_offset = 0.20
    trailing_only_offset_is_reached = True



    # Optimal ticker interval for the strategy.
    timeframe = '15m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    # Optional order type mapping.
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

    plot_config = {
        'main_plot': {
            'tema': {},
            'sar': {'color': 'white'},
        },
        'subplots': {
            "MACD": {
                'macd': {'color': 'blue'},
                'macdsignal': {'color': 'orange'},
            },
            "RSI": {
                'rsi': {'color': 'red'},
            }
        }
    }

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
            """
            Adds several different TA indicators to the given DataFrame

            Performance Note: For the best performance be frugal on the number of indicators
            you are using. Let uncomment only the indicator you are using in your strategies
            or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
            :param dataframe: Dataframe with data from the exchange
            :param metadata: Additional information, like the currently traded pair
            :return: a Dataframe with all mandatory indicators for the strategies
            """

            ichi=ichimoku(dataframe)
            dataframe['tenkan']=ichi['tenkan_sen']
            dataframe['kijun']=ichi['kijun_sen']
            dataframe['senkou_a']=ichi['senkou_span_a']
            dataframe['senkou_b']=ichi['senkou_span_b']
            dataframe['cloud_green']=ichi['cloud_green']
            dataframe['cloud_red']=ichi['cloud_red']
            return dataframe


    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['tenkan'].shift(1)<dataframe['kijun'].shift(1)) &
                (dataframe['tenkan']>dataframe['kijun']) &
                (dataframe['cloud_red']==True)
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
            dataframe.loc[
                (
                ),
                'sell'] = 1
            return dataframe
