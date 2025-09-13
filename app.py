import os
import time
import asyncio
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from telegram import Bot

# -------- CONFIG --------
ALPHA_KEY = os.getenv("ALPHAVANTAGE_KEY")    # API gratuit AlphaVantage
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
PAIRS = os.getenv("PAIRS", "EURUSD,GBPUSD,USDJPY,BTC/USD").split(",")
TF = os.getenv("TF", "5min")
POLL_SEC = int(os.getenv("POLL_SEC", "60"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "60"))
LOOKBACK = int(os.getenv("LOOKBACK", "200"))

# Checks
if not ALPHA_KEY:
    raise ValueError("Setează ALPHAVANTAGE_KEY în variabilele de mediu.")
if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    raise ValueError("Setează TELEGRAM_TOKEN și TELEGRAM_CHAT_ID în variabilele de mediu.")

bot = Bot(token=TELEGRAM_TOKEN)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# -------- Imports optional --------
try:
    import smartmoneyconcepts as smc
except:
    smc = None

try:
    import pytrendline as pt
except:
    pt = None

# -------- AlphaVantage fetch --------
def alpha_fetch_fx_intraday(from_symbol: str, to_symbol: str, interval: str = "5min"):
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "FX_INTRADAY",
        "from_symbol": from_symbol,
        "to_symbol": to_symbol,
        "interval": interval,
        "outputsize": "compact",
        "apikey": ALPHA_KEY,
    }
    r = requests.get(url, params=params, timeout=20)
    if r.status_code != 200:
        logging.warning("AlphaVantage HTTP %s", r.status_code)
        return None
    j = r.json()
    key = next((k for k in j.keys() if "Time Series" in k), None)
    if not key:
        logging.warning("AlphaVantage response unexpected (keys: %s)", list(j.keys())[:6])
        return None
    ts = j[key]
    df = pd.DataFrame.from_dict(ts, orient="index").astype(float)
    df.index = pd.to_datetime(df.index)
    return df.sort_index().tail(LOOKBACK)

def alpha_fetch_crypto_intraday(symbol="BTC", market="USD", interval="5min"):
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "DIGITAL_CURRENCY_INTRADAY",
        "symbol": symbol,
        "market": market,
        "interval": interval,
        "apikey": ALPHA_KEY,
    }
    r = requests.get(url, params=params, timeout=20)
    if r.status_code != 200:
        logging.warning("AlphaVantage HTTP %s", r.status_code)
        return None
    j = r.json()
    data_key = next((k for k in j.keys() if "Time Series" in k), None)
    if not data_key:
        logging.warning("AlphaVantage crypto response unexpected (keys: %s)", list(j.keys())[:6])
        return None
    df = pd.DataFrame(j[data_key]).T
    df = df.rename(columns={
        '1. open (USD)': 'open',
        '2. high (USD)': 'high',
        '3. low (USD)':  'low',
        '4. close (USD)': 'close',
        '5. volume': 'volume'
    }).astype(float)
    df.index = pd.to_datetime(df.index)
    return df.sort_index().tail(LOOKBACK)

# -------- Naive Support/Resistance --------
def naive_pivot_s_r(df: pd.DataFrame, window=20, thresh=0.002):
    highs = df['high']
    lows = df['low']
    piv_high_idx = highs[(highs == highs.rolling(window, center=True).max())].dropna().index
    piv_low_idx = lows[(lows == lows.rolling(window, center=True).min())].dropna().index
    piv_highs = highs.loc[piv_high_idx].values if len(piv_high_idx)>0 else np.array([])
    piv_lows = lows.loc[piv_low_idx].values if len(piv_low_idx)>0 else np.array([])
    zones = []
    for price in piv_highs:
        found = False
        for z in zones:
            if z['type']=='res' and abs(z['price']-price)/price <= thresh:
                z['count'] += 1
                z['price'] = (z['price']*(z['count']-1) + price)/z['count']
                found = True
                break
        if not found:
            zones.append({'type':'res','price':float(price),'count':1})
    for price in piv_lows:
        found = False
        for z in zones:
            if z['type']=='sup' and abs(z['price']-price)/price <= thresh:
                z['count'] += 1
                z['price'] = (z['price']*(z['count']-1) + price)/z['count']
                found = True
                break
        if not found:
            zones.append({'type':'sup','price':float(price),'count':1})
    return sorted(zones, key=lambda x: x['count'], reverse=True)

# -------- BOS (Break of Structure) --------
def simple_bos_detector(df: pd.DataFrame, lookback=50):
    if len(df) < 10:
        return None
    recent = df[-lookback:]
    swing_high = recent['high'].max()
    swing_low = recent['low'].min()
    last_close = df['close'].iloc[-1]
    if last_close > swing_high:
        return {"type":"BOS","direction":"LONG","level":float(swing_high)}
    if last_close < swing_low:
        return {"type":"BOS","direction":"SHORT","level":float(swing_low)}
    return None

# -------- SMC detect --------
def smc_detect(df: pd.DataFrame):
    if smc is None:
        return None
    try:
        ohlc = df[['open','high','low','close']].copy()
        fvg = smc.fvg(ohlc)
        swings = smc.swing_highs_lows(ohlc, swing_length=50)
        bos = smc.bos_choch(ohlc, swings, close_break=True)
        ob = smc.ob(ohlc, swings, close_mitigation=False)
        liq = smc.liquidity(ohlc, swings, range_percent=0.01)
        return {"fvg":fvg, "swings":swings, "bos":bos, "ob":ob, "liquidity":liq}
    except Exception as e:
        logging.exception("SMC detect error: %s", e)
        return None

# -------- Score + signal --------
def score_trade(df_main, df_confirm, zones, bos_info, smc_info):
    score = 0
    direction = None
    last = float(df_confirm['close'].iloc[-1])
    ema20 = df_main['close'].ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = df_main['close'].ewm(span=50, adjust=False).mean().iloc[-1]
    trend = "LONG" if ema20 > ema50 else "SHORT"
    score += 20
    if smc_info and smc_info.get('bos') is not None:
        score += 25
        direction = trend
    if bos_info:
        if bos_info['direction'] == trend:
            score += 25
            direction = trend
        else:
            score += 5
            direction = bos_info['direction']
    if zones:
        top = zones[0]
        dist = abs(last - top['price'])/top['price']
        if dist <= 0.001:
            score += 25
        elif dist <= 0.003:
            score += 15
        elif dist <= 0.006:
            score += 8
        if top['type']=='res' and last > top['price']:
            direction = "LONG"
        if top['type']=='sup' and last < top['price']:
            direction = "SHORT"
    recent_range = (df_confirm['high'] - df_confirm['low']).rolling(10).mean().iloc[-1]
    if recent_range > 0:
        score += min(10, (recent_range / df_confirm['close'].iloc[-1]) * 1000)
    return min(100, round(score,1)), direction or trend

# -------- Telegram send --------
async def send_telegram(msg: str):
    try:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
        logging.info("Telegram sent: %s", msg.splitlines()[0])
    except Exception as e:
        logging.exception("Telegram error: %s", e)

# -------- Analyze pair (FX + CRYPTO) --------
async def analyze_pair(pair: str, sem: asyncio.Semaphore, sent_signals: set):
    async with sem:
        is_crypto = pair.upper() in ["BTC/USD", "BTC/USDT"]

        if is_crypto:
            df_confirm = alpha_fetch_crypto_intraday("BTC", "USD", interval=TF)
            await asyncio.sleep(15)  # respect rate-limit crypto
        else:
            from_sym, to_sym = pair[:3], pair[3:]
            df_confirm = alpha_fetch_fx_intraday(from_sym, to_sym, interval=TF)
            await asyncio.sleep(12)  # respect rate-limit FX

        if df_confirm is None:
            logging.warning("[%s] no data", pair)
            return

        df_main = df_confirm.copy()
        zones = naive_pivot_s_r(df_confirm)
        smc_info = smc_detect(df_confirm)
        bos_info = simple_bos_detector(df_confirm)
        score, direction = score_trade(df_main, df_confirm, zones, bos_info, smc_info)
        ts = df_confirm.index[-1].strftime("%Y-%m-%d %H:%M:%S")
        logging.info("[%s] score=%s dir=%s price=%.5f", pair, score, direction, df_confirm['close'].iloc[-1])
        key = f"{pair}_{ts}_{direction}"
        if score >= MIN_SCORE and key not in sent_signals:
            msg = (
                f"✅ Semnal {direction} ({score}%) – {pair}\n"
                f"TF: {TF}\n"
                f"Preț: {df_confirm['close'].iloc[-1]:.5f}\n"
                f"Time: {ts}\n"
                f"Top zone: {zones[0] if zones else 'none'}\n"
                f"BOS: {bos_info}\n"
            )
            await send_telegram(msg)
            sent_signals.add(key)

# -------- Orchestrator --------
async def main():
    sem = asyncio.Semaphore(2)
    sent_signals = set()
    logging.info("Bot started. Pairs=%s TF=%s", PAIRS, TF)
    while True:
        start = time.time()
        tasks = [analyze_pair(p.strip(), sem, sent_signals) for p in PAIRS]
        await asyncio.gather(*tasks)
        elapsed = time.time() - start
        wait = max(1, POLL_SEC - elapsed)
        await asyncio.sleep(wait)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logging.exception("Fatal: %s", e)
