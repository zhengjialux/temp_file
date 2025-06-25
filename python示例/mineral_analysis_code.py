# 股票数据获取

import tushare as ts
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV


def generate_html_report(industry_stats, stock_data, mineral_news, models_results):
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>矿物制品行业分析报告</title>
        <style>
            body {{font-family:Arial,sans-serif;margin:20px;}}
            h1,h2 {{color:#2e6c80;}}
            table {{border-collapse:collapse;width:100%;margin-bottom:20px;}}
            th,td {{border:1px solid #ddd;padding:8px;text-align:left;}}
            th {{background-color:#f2f2f2;}}
            .plot {{margin:20px 0;}}
            .model-results {{margin-bottom:30px;}}
        </style>
    </head>
    <body>
        <h1>矿物制品行业分析报告</h1>
        
        <h2>1. 行业股票概览</h2>
        {stocks_table}
        
        <h2>2. 价格走势图</h2>
        <div class="plot">
            <img src="price_trend.png" alt="价格走势图" width="800">
        </div>
        
        <h2>3. 相关性分析</h2>
        <div class="plot">
            <img src="correlation_heatmap.png" alt="相关性热力图" width="800">
        </div>
        
        <h2>4. 统计描述</h2>
        {stats_table}

        <h2>5. 层次聚类分析</h2>
        <div class="plot">
            <img src="dendrogram.png" alt="层次聚类图" width="800">
        </div>
        
        <h2>6. 聚类分析结果</h2>
        {clusters_table}
        
        <h2>7. 模型评估结果</h2>
        {models_results}
        
        <h2>8. 行业新闻</h2>
        {news_table}
    </body>
    </html>
    """

    # 生成股票表格
    stocks_table = industry_stats[['name', 'symbol', 'list_date', 'area']].to_html(index=False)
    
    # 生成统计表格
    stats_table = industry_stats[['name', '平均价格', '年收益率', '价格标准差', '最大回撤']].to_html(index=False)
    
    # 生成聚类表格
    clusters_table = industry_stats[['name', '类别名称']].to_html(index=False)
    
    # 生成新闻表格
    news_table = mineral_news.to_html(index=False)
    
    # 生成模型结果
    models_html = ""
    for name, result in models_results.items():
        models_html += f"""
        <div class="model-results">
            <h3>{name}</h3>
            <p>MSE: {result['mse']:.4f}</p>
            <p>R2: {result['r2']:.4f}</p>
        </div>
        """
    
    # 保存图表
    plt.figure(figsize=(12, 6))
    for ts_code, data in stock_data.items():
        plt.plot(pd.to_datetime(data['trade_date']), data['close'], 
                 label=industry_stats.loc[ts_code, 'name'])
    plt.title('矿物制品行业股票价格走势(2023年)', fontsize=14)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('收盘价(元)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid()
    plt.tight_layout()
    plt.savefig('price_trend.png', bbox_inches='tight')
    plt.close()
    
    # 保存热力图
    close_prices = pd.DataFrame()
    for ts_code, data in stock_data.items():
        close_prices[ts_code] = data.set_index('trade_date')['close']
    correlation = close_prices.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', 
                xticklabels=industry_stats['name'], 
                yticklabels=industry_stats['name'])
    plt.title('矿物制品行业股票价格相关性', fontsize=14)
    plt.xticks(fontsize=10, rotation=45)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', bbox_inches='tight')
    plt.close()

    
    # 保存层次聚类图
    plt.figure(figsize=(10, 6))
    dendrogram(linked, 
              labels=industry_stats['name'].tolist(),
              orientation='right')
    plt.title('矿物制品行业股票层次聚类', fontsize=14)
    plt.xlabel('距离', fontsize=12)
    plt.ylabel('股票名称', fontsize=12)
    plt.tight_layout()
    plt.savefig('dendrogram.png', bbox_inches='tight')
    plt.close()
    
    # 填充模板
    final_html = html_template.format(
        stocks_table=stocks_table,
        stats_table=stats_table,
        clusters_table=clusters_table,
        news_table=news_table,
        models_results=models_html
    )
    
    with open('mineral_analysis_report.html', 'w', encoding='utf-8') as f:
        f.write(final_html)
    print("HTML报告已生成: mineral_analysis_report.html")


plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# ----------------------------------------数据可视化 (15%)----------------------------------------
# 初始化Tushare (需要先注册获取token)
pro = ts.pro_api('15339eee5d93d4462091e944621cedc9bfbd53fad47b53b5ff3bc68b')


# ----------------------------------------使用Tushare获取股票数据 (5%)----------------------------------------
# 获取矿物制品行业股票列表
# 拉取数据
df = df = pd.read_csv('tushare_stock_basic_20250618171406.csv')
# df = pro.stock_basic(**{
#     "ts_code": "",
#     "name": "",
#     "exchange": "",
#     "market": "",
#     "is_hs": "",
#     "list_status": "",
#     "limit": "",
#     "offset": ""
# }, fields=[
#     "ts_code",
#     "symbol",
#     "name",
#     "industry",
#     "list_date",
#     "delist_date",
#     "cnspell",
#     "list_status",
#     "area"
# ])
mineral_stocks = df[df['industry'] == '矿物制品'].head(10)  # 取前10支股票

# 获取这些股票的历史行情数据
stock_data = {}
for ts_code in mineral_stocks['ts_code']:
    stock_data[ts_code] = pro.daily(ts_code=ts_code, list_date='20230101', delist_date='20231231')

# 打印Tushare中获取的数据
# print(mineral_stocks)

# ----------------------------------------爬虫获取新闻数据 (10%)----------------------------------------
# 新闻数据爬取
def crawl_mineral_news():
    base_url = "https://search.sina.com.cn/"
    params = {
        'q': '矿物制品',
        'c': 'news',
        'range': 'title',
        'num': 20,
        'page': 1
    }
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    news_data = []
    for page in range(1, 6):
        params['page'] = page
        try:
            response = requests.get(base_url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 更通用的选择器
            for item in soup.find_all('div', class_='box-result'):
                title_element = item.find(['h2', 'h3', 'h4'])
                date_element = item.find(class_=['fgray_time', 'time', 'date'])
                
                title = title_element.get_text(strip=True) if title_element else '无标题'
                date = date_element.get_text(strip=True) if date_element else '无日期'
                
                news_data.append({'title': title, 'date': date})
                
        except Exception as e:
            print(f"第{page}页爬取失败:", str(e))
        
        time.sleep(1)
    
    return pd.DataFrame(news_data)

mineral_news = crawl_mineral_news()
# 打印爬虫到的新闻
# print(mineral_news)



# ----------------------------------------数据可视化 (15%)----------------------------------------
# 绘制矿物制品行业股票价格走势
plt.figure(figsize=(12, 6))
for ts_code, data in stock_data.items():
    plt.plot(pd.to_datetime(data['trade_date']), data['close'], 
             label=mineral_stocks[mineral_stocks['ts_code']==ts_code]['name'].values[0])

plt.title('矿物制品行业股票价格走势(2023年)', fontsize=14)
plt.xlabel('日期', fontsize=12)
plt.ylabel('收盘价(元)', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()
plt.tight_layout()
plt.show()

# 准备相关性分析数据
close_prices = pd.DataFrame()
for ts_code, data in stock_data.items():
    close_prices[ts_code] = data.set_index('trade_date')['close']

correlation = close_prices.corr()

# 绘制热力图
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', 
            xticklabels=mineral_stocks['name'], 
            yticklabels=mineral_stocks['name'])
plt.title('矿物制品行业股票价格相关性', fontsize=14)
plt.xticks(fontsize=10, rotation=45)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()




# ----------------------------------------统计性描述分析 (20%)----------------------------------------

# 行业分析

# 计算各股票的统计指标
stats = pd.DataFrame()
for ts_code, data in stock_data.items():
    stats.loc[ts_code, '平均价格'] = data['close'].mean()
    stats.loc[ts_code, '价格标准差'] = data['close'].std()
    stats.loc[ts_code, '年收益率'] = (data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0]
    stats.loc[ts_code, '最大回撤'] = (data['high'].max() - data['low'].min()) / data['high'].max()

# 合并股票基本信息
industry_stats = mineral_stocks.set_index('ts_code').join(stats)

print_data = industry_stats.copy()
print_data.columns = ['股票名称' if x=='name' else 
                     'TS代码' if x=='ts_code' else 
                     '上市日期' if x=='list_date' else 
                     '上市状态' if x=='list_status' else 
                     '上市地区' if x=='area' else 
                     '所属行业' if x=='industry' else 
                     '股票代码' if x=='symbol' else
                     x for x in print_data.columns]

print("\n矿物制品行业股票统计描述:")
print(print_data.describe())

# 目标股票行业地位分析

# 按市值排序分析行业地位
industry_stats['市值排名'] = industry_stats['平均价格'].rank(ascending=False)
industry_stats['收益率排名'] = industry_stats['年收益率'].rank(ascending=False)
industry_stats['风险排名'] = industry_stats['价格标准差'].rank(ascending=False)

# 同步到print_data
print_data = industry_stats.copy()
print_data.columns = ['股票名称' if x=='name' else 
                     'TS代码' if x=='ts_code' else 
                     '上市日期' if x=='list_date' else 
                     '上市状态' if x=='list_status' else 
                     '上市地区' if x=='area' else 
                     '所属行业' if x=='industry' else 
                     '股票代码' if x=='symbol' else
                     x for x in print_data.columns]

# 输出行业地位分析
print("\n矿物制品行业股票排名分析:")
print(print_data[['股票名称', '平均价格', '年收益率', '价格标准差', '市值排名', '收益率排名', '风险排名']])

# ----------------------------------------数据挖掘分析 (20%)----------------------------------------

# 股票关联性分析

# 层次聚类分析
linked = linkage(correlation, 'ward')  # 添加这行计算linkage矩阵

plt.figure(figsize=(10, 6))
dendrogram(linked, 
           labels=mineral_stocks['name'].tolist(),
           orientation='right')
plt.title('矿物制品行业股票层次聚类', fontsize=14)
plt.xlabel('距离', fontsize=12)
plt.ylabel('股票名称', fontsize=12)
plt.tight_layout()
plt.show()


# 股票聚类分析

# 使用K-means聚类
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(correlation)

# 将聚类结果添加到数据中
industry_stats['聚类类别'] = clusters

# 定义聚类类别名称
cluster_names = {0: '稳健型', 1: '成长型', 2: '波动型'}
industry_stats['类别名称'] = industry_stats['聚类类别'].map(cluster_names)

# 同步到print_data
print_data = industry_stats.copy()
print_data.columns = ['股票名称' if x=='name' else 
                     'TS代码' if x=='ts_code' else 
                     '上市日期' if x=='list_date' else 
                     '上市状态' if x=='list_status' else 
                     '上市地区' if x=='area' else 
                     '所属行业' if x=='industry' else 
                     '股票代码' if x=='symbol' else
                     x for x in print_data.columns]

print("\n股票聚类分析:")
print(print_data[['股票名称', '类别名称']])



# ----------------------------------------建模与模型验证 (30%)----------------------------------------

# 准备数据 - 以一支股票为例
target_stock = list(stock_data.keys())[0]
X = stock_data[target_stock][['open', 'high', 'low', 'vol']]
y = stock_data[target_stock]['close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 随机森林模型
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# 支持向量回归
svr = SVR(kernel='rbf', C=100, gamma=0.1)
svr.fit(X_train, y_train)
svr_pred = svr.predict(X_test)

# 神经网络
mlp = MLPRegressor(hidden_layer_sizes=(100,50), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)
mlp_pred = mlp.predict(X_test)

# 评估模型
models = {
    '随机森林': rf_pred,
    '支持向量机': svr_pred,
    '神经网络': mlp_pred
}

for name, pred in models.items():
    print(f"{name}模型评估:")
    print(f"MSE: {mean_squared_error(y_test, pred):.4f}")
    print(f"R2: {r2_score(y_test, pred):.4f}\n")




# 模型优化与验证
print("\n开始模型优化...")

# 随机森林参数优化
param_grid_rf = {
    'n_estimators': [50, 100, 150],  # 减少参数范围
    'max_depth': [10, 15, 20],      # 限制最大深度
    'min_samples_split': [2, 5]     # 减少参数选项
}

# 添加进度监控
def print_progress(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total: 
        print()

# 优化GridSearchCV设置
grid_rf = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid=param_grid_rf,
    cv=3,  # 减少交叉验证折数
    scoring='r2',
    n_jobs=-1,  # 使用所有CPU核心
    verbose=1  # 显示进度
)

print("正在进行随机森林参数优化...")
grid_rf.fit(X_train, y_train)

print("\n优化完成!")
print("最优随机森林参数:", grid_rf.best_params_)
print("最优模型R2分数:", grid_rf.best_score_)

# 使用最优模型进行预测
best_rf = grid_rf.best_estimator_
best_pred = best_rf.predict(X_test)
print("测试集R2:", r2_score(y_test, best_pred))

models_results = {
    '随机森林': {'mse': mean_squared_error(y_test, rf_pred), 'r2': r2_score(y_test, rf_pred)},
    '支持向量机': {'mse': mean_squared_error(y_test, svr_pred), 'r2': r2_score(y_test, svr_pred)},
    '神经网络': {'mse': mean_squared_error(y_test, mlp_pred), 'r2': r2_score(y_test, mlp_pred)},
    '优化后随机森林': {'mse': mean_squared_error(y_test, best_pred), 'r2': r2_score(y_test, best_pred)}
}

# 生成HTML报告
generate_html_report(industry_stats, stock_data, mineral_news, models_results)