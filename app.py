# app.py
import os
import time
import asyncio
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from telegram import Bot

# -------- CONFIG (mediu) --------
ALPHA_KEY = os.getenv("ALPHAVANTAGE_KEY")    # required
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
PAIRS = os.getenv("PAIRS", "EURUSD,GBPUSD,USDJPY").split(",")
TF = os.getenv("TF", "5min")                 # AlphaVantage intervals: 1min,5min,15min...
POLL_SEC = int(os.getenv("POLL_SEC", "60"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "60"))
LOOKBACK = int(os.getenv("LOOKBACK", "200"))
AV_RATE_LIMIT_SEC = 12   # AlphaVantage free rate ~5 req/min => ~12s between calls

# basic checks
if not ALPHA_KEY:
    raise ValueError("Setează ALPHAVANTAGE_KEY în variabilele de mediu.")
if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    raise ValueError("Setează TELEGRAM_TOKEN și TELEGRAM_CHAT_ID în variabilele de mediu.")

bot = Bot(token=TELEGRAM_TOKEN)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# -------- try imports (optional libs) --------
try:
    import smartmoneyconcepts as smc   # pip install smartmoneyconcepts
except Exception:
    smc = None

try:
    import support_resistance as sr    # optional, from GitHub
except Exception:
    sr = None

try:
    import pytrendline as pt           # optional, from GitHub
except Exception:
    pt = None

# -------- AlphaVantage fetch for FX_INTRADAY --------
def alpha_fetch_fx_intraday(from_symbol: str, to_symbol: str, interval: str = "5min", outputsize: str = "compact"):
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "FX_INTRADAY",
        "from_symbol": from_symbol,
        "to_symbol": to_symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": ALPHA_KEY,
    }
    r = requests.get(url, params=params, timeout=20)
    if r.status_code != 200:
        logging.warning("AlphaVantage HTTP %s", r.status_code)
        return None
    j = r.json()
    # find time series key
    key = next((k for k in j.keys() if "Time Series" in k), None)
    if not key:
        logging.warning("AlphaVantage response unexpected (keys: %s)", list(j.keys())[:6])
        return None
    ts = j[key]
    df = pd.DataFrame.from_dict(ts, orient="index")
    df.columns = ["open", "high", "low", "close"]
    df = df.astype(float)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df.tail(LOOKBACK)

# -------- Fallback Support/Resistance (naive pivot clustering) --------
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
    zones = sorted(zones, key=lambda x: x['count'], reverse=True)
    return zones

# -------- Simple BOS fallback (naive) --------
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

# -------- SMC helpers using smartmoneyconcepts (if available) --------
def smc_detect(df: pd.DataFrame):
    """
    Returns dict with keys: 'fvg', 'swing', 'bos', 'ob', 'liquidity' if smc present,
    else None
    """
    out = {}
    try:
        if smc is None:
            return None
        # smc expects lowercase columns
        ohlc = df[['open','high','low','close']].copy()
        fvg = smc.fvg(ohlc)                     # returns array-like with FVG flags
        swings = smc.swing_highs_lows(ohlc, swing_length=50)
        bos = smc.bos_choch(ohlc, swings, close_break=True)
        ob = smc.ob(ohlc, swings, close_mitigation=False)
        liq = smc.liquidity(ohlc, swings, range_percent=0.01)
        out.update({"fvg":fvg, "swing":swings, "bos":bos, "ob":ob, "liquidity":liq})
        return out
    except Exception as e:
        logging.exception("SMC detect error: %s", e)
        return None

# -------- Scoring combine S/R + BOS + trend + volatility --------
def score_trade(df_main: pd.DataFrame, df_confirm: pd.DataFrame, zones, bos_info, smc_info):
    score = 0
    direction = None
    last = float(df_confirm['close'].iloc[-1])

    # trend (EMA20 vs EMA50)
    ema20 = df_main['close'].ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = df_main['close'].ewm(span=50, adjust=False).mean().iloc[-1]
    trend = "LONG" if ema20 > ema50 else "SHORT"
    score += 20

    # SMC bos_info preference
    if smc_info and 'bos' in smc_info and smc_info['bos'] is not None:
        # smc returns arrays -> we'll do a simple score boost
        score += 25
        direction = trend

    # fallback bos
    if bos_info:
        if bos_info['direction'] == trend:
            score += 25
            direction = trend
        else:
            score += 5
            direction = bos_info['direction']

    # proximity to zone
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

    # volatility (range)
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

# -------- analyze per pair --------
async def analyze_pair(pair: str, sem: asyncio.Semaphore, sent_signals: set):
    async with sem:
        from_sym = pair[:3]
        to_sym = pair[3:]
        # fetch confirm timeframe df
        df_confirm = alpha_fetch_fx_intraday(from_sym, to_sym, interval=TF)
        if df_confirm is None:
            logging.warning("[%s] no data", pair)
            return
        # use same as main for trend — can change to larger TF
        df_main = df_confirm.copy()

        # zones: try library first, fallback to naive
        zones = None
        if sr is not None:
            try:
                # NOTE: API of support_resistance may differ; adapt if needed
                zones = sr.find_support_resistance(df_confirm)  # pseudo-call (adjust if lib differs)
            except Exception:
                zones = naive_pivot_s_r(df_confirm)
        else:
            zones = naive_pivot_s_r(df_confirm)

        # SMC
        smc_info = smc_detect(df_confirm)

        # BOS
        bos_info = None
        if smc_info and 'bos' in smc_info and smc_info['bos'] is not None:
            bos_info = {'type':'BOS', 'direction': "LONG" if smc_info['bos'][0] == 1 else "SHORT", 'level': None}
        else:
            bos_info = simple_bos_detector(df_confirm)

        # score
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

# -------- orchestrator --------
async def main():
    sem = asyncio.Semaphore(2)
    sent_signals = set()
    logging.info("Bot started. Pairs=%s TF=%s", PAIRS, TF)
    while True:
        start = time.time()
        tasks = []
        for p in PAIRS:
            tasks.append(analyze_pair(p.strip(), sem, sent_signals))
            await asyncio.sleep(AV_RATE_LIMIT_SEC)  # throttle for AlphaVantage
        await asyncio.gather(*tasks)
        elapsed = time.time() - start
        wait = max(1, POLL_SEC - elapsed)
        await asyncio.sleep(wait)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logging.exception("Fatal: %s", e)
