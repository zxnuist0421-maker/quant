#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股量化交易Demo
功能：
1. 获取A股股票数据
2. 计算技术指标
3. 可视化结果
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import akshare as ak
import talib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class AShareQuantStrategy:
    def __init__(self, symbol='600519', start_date=None, end_date=None):
        """
        初始化A股策略
        :param symbol: A股代码（如：'600519' 贵州茅台）
        :param start_date: 开始日期
        :param end_date: 结束日期
        """
        self.symbol = symbol
        self.start_date = start_date or (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.data = None
        self.stock_name = None
        
    def fetch_data(self):
        """获取A股股票数据"""
        print(f"正在获取 {self.symbol} 的数据...")
        try:
            # 获取A股数据（前复权）
            start_date_format = self.start_date.replace('-', '')
            end_date_format = self.end_date.replace('-', '')
            
            self.data = ak.stock_zh_a_hist(
                symbol=self.symbol, 
                start_date=start_date_format, 
                end_date=end_date_format, 
                adjust="qfq"  # 前复权
            )
            
            if self.data.empty:
                print("获取的数据为空，请检查股票代码或日期范围")
                return False
                
            # 设置日期为索引
            self.data['日期'] = pd.to_datetime(self.data['日期'])
            self.data.set_index('日期', inplace=True)
            
            # 重命名列名为英文，方便后续处理
            column_mapping = {
                '开盘': 'Open',
                '收盘': 'Close', 
                '最高': 'High',
                '最低': 'Low',
                '成交量': 'Volume',
                '成交额': 'Amount',
                '振幅': 'Amplitude',
                '涨跌幅': 'Change_pct',
                '涨跌额': 'Change_amount',
                '换手率': 'Turnover'
            }
            self.data.rename(columns=column_mapping, inplace=True)
            
            # 获取股票名称
            try:
                stock_info = ak.stock_individual_info_em(symbol=self.symbol)
                self.stock_name = stock_info[stock_info['item'] == '股票简称']['value'].iloc[0]
            except:
                self.stock_name = self.symbol
                
            print(f"成功获取 {self.stock_name}({self.symbol}) {len(self.data)} 条数据")
            return True
            
        except Exception as e:
            print(f"获取数据失败: {e}")
            print("建议：检查网络连接和股票代码格式，如：'600519'（贵州茅台）")
            return False
    
    def calculate_indicators(self):
        """计算技术指标"""
        if self.data is None:
            print("请先获取数据")
            return
            
        # 移动平均线
        self.data['MA5'] = talib.SMA(self.data['Close'].values, timeperiod=5)
        self.data['MA10'] = talib.SMA(self.data['Close'].values, timeperiod=10)
        self.data['MA20'] = talib.SMA(self.data['Close'].values, timeperiod=20)
        self.data['MA60'] = talib.SMA(self.data['Close'].values, timeperiod=60)
        
        # RSI
        self.data['RSI'] = talib.RSI(self.data['Close'].values, timeperiod=14)
        
        # MACD
        macd, macdsignal, macdhist = talib.MACD(self.data['Close'].values)
        self.data['MACD'] = macd
        self.data['MACD_Signal'] = macdsignal
        self.data['MACD_Hist'] = macdhist
        
        # 布林带
        upper, middle, lower = talib.BBANDS(self.data['Close'].values, timeperiod=20)
        self.data['BB_Upper'] = upper
        self.data['BB_Middle'] = middle
        self.data['BB_Lower'] = lower
        
        # KDJ指标
        high = self.data['High'].values
        low = self.data['Low'].values 
        close = self.data['Close'].values
        
        k, d = talib.STOCH(high, low, close, fastk_period=9, slowk_period=3, slowd_period=3)
        self.data['K'] = k
        self.data['D'] = d
        self.data['J'] = 3 * k - 2 * d
        
        print("技术指标计算完成")
    
    def plot_results(self):
        """可视化结果"""
        if self.data is None:
            return
            
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        
        # 价格和移动平均线
        axes[0].plot(self.data.index, self.data['Close'], label='收盘价', linewidth=1.5, color='black')
        axes[0].plot(self.data.index, self.data['MA5'], label='MA5', alpha=0.8, color='red')
        axes[0].plot(self.data.index, self.data['MA10'], label='MA10', alpha=0.8, color='orange')
        axes[0].plot(self.data.index, self.data['MA20'], label='MA20', alpha=0.8, color='blue')
        axes[0].plot(self.data.index, self.data['MA60'], label='MA60', alpha=0.8, color='green')
        
        # 布林带
        axes[0].fill_between(self.data.index, self.data['BB_Upper'], self.data['BB_Lower'], 
                           alpha=0.1, color='gray', label='布林带')
        
        axes[0].set_title(f'{self.stock_name}({self.symbol}) 价格走势和移动平均线', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # RSI
        axes[1].plot(self.data.index, self.data['RSI'], label='RSI', color='purple', linewidth=1.5)
        axes[1].axhline(y=70, color='r', linestyle='--', alpha=0.7, label='超买线(70)')
        axes[1].axhline(y=30, color='g', linestyle='--', alpha=0.7, label='超卖线(30)')
        axes[1].axhline(y=50, color='gray', linestyle='-', alpha=0.5)
        axes[1].set_title('RSI相对强弱指标', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 100)
        
        # MACD
        axes[2].plot(self.data.index, self.data['MACD'], label='MACD', color='blue', linewidth=1.5)
        axes[2].plot(self.data.index, self.data['MACD_Signal'], label='信号线', color='red', linewidth=1.5)
        
        # MACD柱状图
        colors = ['red' if x > 0 else 'green' for x in self.data['MACD_Hist']]
        axes[2].bar(self.data.index, self.data['MACD_Hist'], color=colors, alpha=0.6, label='MACD柱')
        
        axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[2].set_title('MACD指标', fontsize=12)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # KDJ指标
        axes[3].plot(self.data.index, self.data['K'], label='K线', color='blue', linewidth=1.5)
        axes[3].plot(self.data.index, self.data['D'], label='D线', color='red', linewidth=1.5)
        axes[3].plot(self.data.index, self.data['J'], label='J线', color='orange', linewidth=1.5)
        axes[3].axhline(y=80, color='r', linestyle='--', alpha=0.7, label='超买线(80)')
        axes[3].axhline(y=20, color='g', linestyle='--', alpha=0.7, label='超卖线(20)')
        axes[3].set_title('KDJ指标', fontsize=12)
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        axes[3].set_ylim(0, 100)
        
        plt.tight_layout()
        plt.show()
    
    def show_data_summary(self):
        """显示数据摘要"""
        if self.data is None:
            return
            
        print(f"\n=== {self.stock_name}({self.symbol}) 数据摘要 ===")
        print(f"数据时间范围: {self.data.index[0].strftime('%Y-%m-%d')} 到 {self.data.index[-1].strftime('%Y-%m-%d')}")
        print(f"总交易日数: {len(self.data)} 天")
        
        latest = self.data.iloc[-1]
        print(f"\n最新数据 ({self.data.index[-1].strftime('%Y-%m-%d')}):")
        print(f"收盘价: {latest['Close']:.2f}")
        print(f"涨跌幅: {latest['Change_pct']:.2f}%")
        print(f"成交量: {latest['Volume']:,}")
        
        print(f"\n技术指标:")
        print(f"MA5: {latest['MA5']:.2f}")
        print(f"MA20: {latest['MA20']:.2f}")
        print(f"RSI: {latest['RSI']:.2f}")
        print(f"MACD: {latest['MACD']:.4f}")
        
        # 简单的技术分析提示
        print(f"\n简单技术分析:")
        if latest['Close'] > latest['MA5'] > latest['MA20']:
            print("• 价格位于均线之上，短期趋势向上")
        elif latest['Close'] < latest['MA5'] < latest['MA20']:
            print("• 价格位于均线之下，短期趋势向下")
        else:
            print("• 价格与均线交织，趋势不明")
            
        if latest['RSI'] > 70:
            print("• RSI超买，注意回调风险")
        elif latest['RSI'] < 30:
            print("• RSI超卖，可能有反弹机会")
        else:
            print(f"• RSI在正常区间({latest['RSI']:.1f})")
    
    def run_analysis(self):
        """运行完整分析"""
        print("=== A股量化分析开始 ===")
        
        # 1. 获取数据
        if not self.fetch_data():
            return None
            
        # 2. 计算指标
        self.calculate_indicators()
        
        # 3. 显示摘要
        self.show_data_summary()
        
        # 4. 可视化
        self.plot_results()
        
        print("分析完成！")
        return self.data


def main():
    """主函数 - 运行A股Demo"""
    print("=== A股量化分析Demo ===")
    
    # 一些常见的A股代码示例
    popular_stocks = {
        '600519': '贵州茅台',
        '000858': '五粮液', 
        '600036': '招商银行',
        '000001': '平安银行',
        '600000': '浦发银行',
        '600276': '恒瑞医药',
        '600887': '伊利股份'
    }
    
    print("热门股票代码:")
    for code, name in popular_stocks.items():
        print(f"  {code}: {name}")
    
    # 默认分析贵州茅台
    symbol = '600519'  # 可以修改为其他股票代码
    
    # 创建策略实例
    strategy = AShareQuantStrategy(
        symbol=symbol,
        start_date='2023-01-01'
    )
    
    # 运行分析
    results = strategy.run_analysis()
    
    if results is not None:
        print(f"\n=== 最近5天数据 ===")
        columns_to_show = ['Close', 'MA5', 'MA20', 'RSI', 'Change_pct']
        print(results[columns_to_show].tail())


if __name__ == "__main__":
    main()
    
