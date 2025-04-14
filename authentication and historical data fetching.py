import upstox_client
from upstox_client.rest import ApiException
from upstox_client.api.historical_candle_data_api import HistoricalCandleDataApi

api_key = 'your_api_key'
api_secret = 'your_api_secret'
session_token = 'your_session_token'

configuration = upstox_client.Configuration()
configuration.api_key = api_key
configuration.api_secret = api_secret
configuration.session_token = session_token
api_client = upstox_client.ApiClient(configuration)

api_instance = HistoricalCandleDataApi(api_client)
instrument_token = "NSE_INDEX|Nifty 50"  # Adjust if necessary
interval = "minute"
start_date = (pd.to_datetime('2025-04-14') - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
end_date = '2025-04-14'

try:
    api_response = api_instance.get_historical_candle_data(instrument_token=instrument_token, interval=interval, start_date=start_date, end_date=end_date)
    historical_data = api_response['data']  # Adjust based on actual response structure
    df = pd.DataFrame(historical_data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    close_prices = df['close'].values.reshape(-1, 1)
except ApiException as e:
    print("Exception when calling HistoricalCandleDataApi->get_historical_candle_data: %s\n" % e)
    close_prices = None
