import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as st

# STEP 1. Data preprocessing
# Visits table
visits = pd.read_csv('visits_log_us.csv',
                     dtype={'Device': 'category'},  # Changing object to category
                     parse_dates=['Start Ts', 'End Ts'])  # Changing object to datetime
# Column names to lowercase
visits.columns = visits.columns.str.lower()
# Just adding '_' instead of spaces
visits.rename(columns={'start ts': 'start_ts', 'end ts': 'end_ts', 'source id': 'source_id'}, inplace=True)
print(visits.head())
print()
visits.info(memory_usage='deep')
print()

# Orders table
orders = pd.read_csv('orders_log_us.csv',
                     parse_dates=['Buy Ts'])  # Changing object to datetime
# Column names to lowercase
orders.columns = orders.columns.str.lower()
# Adding '_'
orders.rename(columns={'buy ts': 'buy_ts'}, inplace=True)
print(orders.head())
print()
print(orders.info(memory_usage='deep'))
print()

# Costs table
costs = pd.read_csv('costs_us.csv',
                    parse_dates=['dt'])  # Changing object to datetime
print(costs.head())
print()
costs.info(memory_usage='deep')
print()

# STEP 2. Calculations
# Adding new required columns
visits['session_year'] = visits['start_ts'].dt.year
visits['session_month'] = visits['start_ts'].dt.month
visits['session_week'] = visits['start_ts'].dt.week
visits['session_date'] = visits['start_ts'].dt.date
# print(visits.head())
# print()

# Calculating DAU
print('Daily unique users total:')
dau_total = visits.groupby('session_date').agg({'uid': 'nunique'}).mean()
print(int(dau_total))
print()

# Calculating WAU
print('Weekly unique users total:')
wau_total = visits.groupby('session_week').agg({'uid': 'nunique'}).mean()
print(int(wau_total))
print()

# Calculating MAU
print('Monthly unique users total:')
mau_total = visits.groupby('session_month').agg({'uid': 'nunique'}).mean()
print(int(mau_total))
print()

# Calculating sticky factors
print('Sticky factors (%)')
sticky_wau = dau_total / wau_total * 100
print(sticky_wau)
sticky_mau = dau_total / mau_total * 100
print(sticky_mau)
print()

# Number of sessions per day
sessions_per_user = visits.groupby('session_date').agg({'uid': ['count', 'nunique']})
sessions_per_user.columns = ['n_sessions', 'n_users']
sessions_per_user['sessions_per_user'] = sessions_per_user['n_sessions'] / sessions_per_user['n_users']
print(sessions_per_user)
print()
ax1 = sessions_per_user['n_sessions'].hist(bins=100, figsize=(7, 5))
ax1.set_title('Number of sessions')
plt.show()
print('Average number of sessions per day:')
print(sessions_per_user['n_sessions'].mean())
print()

# Calculating length of sessions
visits['session_duration_sec'] = (visits['end_ts'] - visits['start_ts']).dt.seconds
print(visits[['uid', 'session_duration_sec']])
ax2 = visits['session_duration_sec'].hist(bins=100, figsize=(7, 5))
ax2.set_title('Session duration')
plt.show()
print('Average session duration in seconds:')
print(float(visits['session_duration_sec'].mode()))
print('Number of sessions with 0 sec duration')
print(visits.query('session_duration_sec == 0')['session_duration_sec'].count())

# sales info
# Capturing month from dates
orders['buy_month'] = orders['buy_ts'].astype('datetime64[M]')
print('First purchase')
print(orders['buy_ts'].min())
print('Number of month:')
print(orders['buy_month'].value_counts().count())
print()
print('Total revenue:')
print(orders['revenue'].sum())
print()

# Creating new table with first user's purchase
first_orders = orders.groupby('uid').agg({'buy_month': 'min'}).reset_index()
first_orders.columns = ['uid', 'first_buy_month']

# Calculating the number of new buyers for each month
month_buyers = first_orders.groupby('first_buy_month').agg({'uid': 'nunique'}).reset_index()
month_buyers.columns = ['first_buy_month', 'n_buyers']
print(month_buyers)
month_buyers.plot(kind='barh', figsize=(9, 7), x='first_buy_month')
plt.show()
print()

# Joining orders and first order for each user
orders_ = pd.merge(orders, first_orders, on='uid')
print(orders_)
print()

# Building cohort
cohorts = orders_.groupby(['first_buy_month', 'buy_month']).agg({'revenue': 'sum'}).reset_index()
print(cohorts)
print()

# Adding cohort age and ltv to print result table
report = pd.merge(cohorts, month_buyers, on='first_buy_month')
report['age'] = ((report['buy_month'] - report['first_buy_month']) / np.timedelta64(1, 'M')).round().astype('int')
report['ltv'] = report['revenue'] / report['n_buyers']
result = report.pivot_table(
    index='first_buy_month',
    columns='age',
    values='ltv',
    aggfunc='mean').round(2).fillna('')
print(result)
print()

# Time between first visit and first purchase, amount of orders
first_visits = visits.groupby('uid').agg({'start_ts': 'min'}).reset_index()
first_visits.columns = ['uid', 'first_start_ts']
first_buys = orders.groupby('uid').agg({'buy_ts': 'min'}).reset_index()
first_buys.columns = ['uid', 'first_buy_ts']
user = pd.merge(first_visits, first_buys, on='uid')
user['length'] = user['first_buy_ts'] - user['first_start_ts']

buys = orders.groupby('uid').agg({'buy_ts': 'count'}).reset_index()
buys.columns = ['uid', 'n_orders']
user = user.merge(buys, on='uid')
print(user)
print(user['n_orders'].mean())
print(user['length'].mean())

# marketing info
# Capturing month from dates
costs['dt_month'] = costs['dt'].astype('datetime64[M]')

# Info about money
print('Total money spent:')
total_marketing = costs['costs'].sum()
print(total_marketing)
print()
print('Money spent by each source:')
costs_source = costs.pivot_table(index='source_id', values='costs', aggfunc='sum')
print(costs_source)
print()
costs_source.plot(kind='barh', figsize=(9, 7))
plt.show()
print('Money spent by each month:')
costs_month = costs.pivot_table(index='dt_month', values='costs', aggfunc='sum')
print(costs_month)
costs_month.plot(kind='barh', figsize=(9, 7))
plt.show()

# Calculating number of users for each source
user2 = visits[['uid', 'source_id']].copy()
source_users = user2.groupby('source_id').agg({'uid': 'count'})
source_users.columns = ['n_users']
costs_source = costs_source.merge(source_users, on='source_id', how='left')

# Calculating CAC
print('CAC:')
costs_source['cac'] = costs_source['costs'] / costs_source['n_users']

# Calculating LTV for each source
order_revenue = orders[['uid', 'revenue']].copy()
order_source = order_revenue.merge(user2, on='uid')
order_source = order_source.drop_duplicates()
source_revenue = order_source.groupby('source_id').agg({'revenue': 'sum'})
source_revenue.columns = ['total_revenue']
costs_source = costs_source.merge(source_revenue, on='source_id', how='left')
costs_source['ltv'] = costs_source['total_revenue'] / costs_source['n_users']

# Calculating ROMI
costs_source['romi'] = costs_source['ltv'] / costs_source['cac']
print(costs_source)
