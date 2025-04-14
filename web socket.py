from upstox_client import MarketDataStreamer

streamer = MarketDataStreamer(api_client)
instruments = ['NSE_INDEX|Nifty 50']  # Example
live_data = []  # To store recent live prices

def on_message(message):
    global live_data
    live_data.append(float(message['data']['last_price']))  # Adjust based on actual structure
    if len(live_data) > 60:
        live_data.pop(0)

streamer.subscribe(instruments)
streamer.on('message', on_message)
