import os
import time
import pandas as pd
import numpy as np
from tvDatafeed import TvDatafeed, Interval

# ============================================================
# CONFIGURAZIONE FILE STATICI
# ============================================================
INPUT_FILE = "Tadingview_SPNQDW_MIBDAX_assets.txt"
BODY_THRESHOLD = 70.0

# Ricava il nome base senza estensione per generare il CSV di uscita
if os.path.exists(INPUT_FILE):
    title_label = os.path.splitext(os.path.basename(INPUT_FILE))[0]
    out_csv = f"{title_label}_drawdown_stats.csv"
else:
    raise FileNotFoundError(f"Il file di input obbligatorio '{INPUT_FILE}' non è stato trovato nella cartella corrente.")

# ============================================================
# CONNESSIONE TRADINGVIEW
# ============================================================
tv = TvDatafeed()

# ============================================================
# LETTURA TICKER / EXCHANGE
# ============================================================
with open(INPUT_FILE, 'r') as f:
    lines = [l.strip() for l in f.readlines() if l.strip()]

tickers_exchange = []
for line in lines:
    parts = line.split()
    if len(parts) >= 2:
        tickers_exchange.append((parts[0].upper(), parts[1].upper()))
    else:
        print(f"  [SKIP] Riga non valida: '{line}'")

print(f"\nTicker caricati: {len(tickers_exchange)}")

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def get_max_history_tv(symbol: str, exchange: str, bars_per_call=5000, max_retries=3, timeout_sec=60):
    prev_len = 0
    n_bars   = bars_per_call

    for attempt in range(max_retries + 1):
        try:
            # Gestione del timeout sicura anche per ambienti non-Linux (es. Windows locale)
            try:
                import signal
                def _timeout_handler(signum, frame):
                    raise TimeoutError(f"Timeout dopo {timeout_sec}s")
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(timeout_sec)
            except AttributeError:
                pass # signal.SIGALRM non presente su Windows

            df = tv.get_hist(
                symbol=symbol,
                exchange=exchange,
                interval=Interval.in_daily,
                n_bars=n_bars
            )

            try:
                signal.alarm(0) # Disattiva alarm se supportato
            except NameError:
                pass

            if df is None or df.empty:
                return pd.DataFrame()

            curr_len = len(df)
            if curr_len <= prev_len or curr_len < n_bars:
                break
            prev_len = curr_len
            n_bars  += bars_per_call

        except TimeoutError as e:
            print(f"    ⏱ {symbol}: {e} — skip")
            return pd.DataFrame()
        except Exception as e:
            if attempt < max_retries:
                wait = 3 * (attempt + 1)
                time.sleep(wait)
            else:
                print(f"    ✗ {symbol}: fallito dopo {max_retries} tentativi — skip")
                return pd.DataFrame()

    if 'df' not in locals() or df is None or df.empty:
        return pd.DataFrame()

    try:
        df_out = df[['close']].copy()
        df_out.columns = ['Price']
        df_out.index = pd.to_datetime(df_out.index).tz_localize(None)
        if all(c in df.columns for c in ['high', 'low', 'volume']):
            df_out['High']   = df['high'].values
            df_out['Low']    = df['low'].values
            df_out['Volume'] = df['volume'].values
        if all(c in df.columns for c in ['open', 'close', 'high', 'low']):
            df_out['Open']  = df['open'].values
            df_out['Close'] = df['close'].values
        return df_out.sort_index()
    except Exception as e:
        print(f"    ✗ {symbol}: errore formattazione dati: {e}")
        return pd.DataFrame()

def get_weekly_history_tv(symbol: str, exchange: str, n_bars: int = 100, max_retries: int = 3, timeout_sec: int = 60):
    for attempt in range(max_retries + 1):
        try:
            try:
                import signal
                def _timeout_handler(signum, frame):
                    raise TimeoutError(f"Timeout weekly dopo {timeout_sec}s")
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(timeout_sec)
            except AttributeError:
                pass

            df = tv.get_hist(
                symbol=symbol,
                exchange=exchange,
                interval=Interval.in_weekly,
                n_bars=n_bars
            )

            try:
                signal.alarm(0)
            except NameError:
                pass

            if df is None or df.empty:
                return pd.DataFrame()

            df_out = pd.DataFrame(index=pd.to_datetime(df.index).tz_localize(None))
            df_out['Open']  = df['open'].values
            df_out['High']  = df['high'].values
            df_out['Low']   = df['low'].values
            df_out['Close'] = df['close'].values
            return df_out.sort_index()

        except TimeoutError as e:
            print(f"    ⏱ {symbol} weekly: {e} — skip")
            return pd.DataFrame()
        except Exception as e:
            if attempt < max_retries:
                wait = 3 * (attempt + 1)
                time.sleep(wait)
            else:
                return pd.DataFrame()
    return pd.DataFrame()

def calc_body_pct(df_ohlc: pd.DataFrame, bar_index: int = -2):
    try:
        if df_ohlc is None or df_ohlc.empty or len(df_ohlc) < abs(bar_index):
            return None, 'N/A'
        row   = df_ohlc.iloc[bar_index]
        h, l  = float(row['High']), float(row['Low'])
        o, c  = float(row['Open']), float(row['Close'])
        total = h - l
        if total <= 0:
            return None, 'N/A'
        body_pct = abs(c - o) / total * 100.0
        flag     = 'YES' if body_pct >= BODY_THRESHOLD else 'NO'
        return round(body_pct, 1), flag
    except Exception:
        return None, 'N/A'

def calculate_single_poc(data, num_bins=50):
    try:
        if not all(c in data.columns for c in ['High', 'Low', 'Volume']):
            return None
        price_min = data['Low'].min()
        price_max = data['High'].max()
        if price_min == price_max:
            return None
        bins = np.linspace(price_min, price_max, num_bins)
        volume_profile = np.zeros(num_bins - 1)
        for _, row in data.iterrows():
            low, high, volume = row['Low'], row['High'], row['Volume']
            for i in range(len(bins) - 1):
                if high >= bins[i] and low <= bins[i + 1]:
                    volume_profile[i] += volume
        poc_index = np.argmax(volume_profile)
        poc_price = (bins[poc_index] + bins[poc_index + 1]) / 2
        return round(poc_price, 2)
    except Exception:
        return None

def calculate_three_pocs(df):
    try:
        max_idx     = df['Price'].idxmax()
        df_to_max   = df[df.index <= max_idx]
        df_from_max = df[df.index >= max_idx]
        poc1 = calculate_single_poc(df_to_max)
        poc2 = calculate_single_poc(df_from_max)
        poc3 = calculate_single_poc(df)
        return poc1, poc2, poc3
    except Exception:
        return None, None, None

def build_tv_link(exchange, ticker):
    return f"https://www.tradingview.com/chart/?symbol={exchange}%3A{ticker}&interval=W"

def get_market_cap(symbol: str, exchange: str) -> float | None:
    try:
        import yfinance as yf
        exchange_suffix = {
            'LSE': '.L', 'XETR': '.DE', 'EURONEXT': '.PA', 'BVMF': '.SA',
            'TSX': '.TO', 'ASX': '.AX', 'TYO': '.T', 'HKEX': '.HK',
            'NSE': '.NS', 'BSE': '.BO', 'MIL': '.MI', 'STO': '.ST',
            'EPA': '.PA', 'BME': '.MC', 'AMS': '.AS', 'SWX': '.SW',
        }
        suffix = exchange_suffix.get(exchange.upper(), '')
        ticker_yf = symbol + suffix
        info = yf.Ticker(ticker_yf).info
        mc = info.get('marketCap') or info.get('market_cap')
        return float(mc) if mc else None
    except Exception:
        return None

# ============================================================
# PROCESSING LOOP
# ============================================================
results = []
n_total = len(tickers_exchange)

print(f"\nElaborazione {n_total} ticker...")

for i, (symbol, exchange) in enumerate(tickers_exchange, 1):
    print(f"[{i:3d}/{n_total}] {symbol:10s} ({exchange})", end=" → ", flush=True)

    if i > 1:
        time.sleep(1.5)

    try:
        df = get_max_history_tv(symbol, exchange)
        if df.empty:
            print("SKIP (nessun dato)")
            continue

        body_d_pct, body_d_flag = calc_body_pct(df, bar_index=-2)

        df_w = get_weekly_history_tv(symbol, exchange, n_bars=100)
        body_w_pct, body_w_flag = calc_body_pct(df_w, bar_index=-2)

        market_cap = get_market_cap(symbol, exchange)

        years = df.index.year.unique()
        yearly_mdd = []

        for year in years:
            yd = df[df.index.year == year].copy()
            if len(yd) < 20:
                continue
            yd['Peak']     = yd['Price'].cummax()
            yd['Drawdown'] = (yd['Price'] - yd['Peak']) / yd['Peak']
            yearly_mdd.append(abs(yd['Drawdown'].min() * 100))

        if not yearly_mdd:
            print("SKIP (dati insufficienti)")
            continue

        dd_medio    = np.mean(yearly_mdd)
        dd_massimo  = np.max(yearly_mdd)
        durata_anni = len(yearly_mdd)

        price_start   = float(df['Price'].iloc[0])
        current_price = float(df['Price'].iloc[-1])
        n_years       = (df.index[-1] - df.index[0]).days / 365.25
        cagr = ((current_price / price_start) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0

        poc1, poc2, poc3 = calculate_three_pocs(df)

        if poc1 is not None and poc2 is not None and poc3 is not None:
            sopra_3poc = "YES" if (current_price > poc1 and current_price > poc2 and current_price > poc3) else "NO"
        else:
            sopra_3poc = "N/A"

        tv_link = build_tv_link(exchange, symbol)

        results.append({
            'Ticker':        symbol,
            'Exchange':      exchange,
            'DD_Medio':      -dd_medio,
            'Max_DD':        -dd_massimo,
            'Durata_Anni':   durata_anni,
            'CAGR':          cagr,
            'Prezzo':        round(current_price, 2),
            'POC1':          poc1,
            'POC2':          poc2,
            'POC3':          poc3,
            'Sopra_3POC':    sopra_3poc,
            'Body_D%':       body_d_pct,
            'Body_D_Flag':   body_d_flag,
            'Body_W%':       body_w_pct,
            'Body_W_Flag':   body_w_flag,
            'Link_TV_W':     tv_link,
            'Market_Cap':    market_cap,
        })
        print("OK")

    except Exception as e:
        print(f"ERR ({e})")

# ============================================================
# SALVATAGGIO CSV ESATTO E FINALE
# ============================================================
if len(results) == 0:
    raise RuntimeError("Nessun ticker elaborato con successo. Controlla il file di input.")

res_df = pd.DataFrame(results)

# Genera esattamente lo stesso CSV strutturato con round(3), separatore ';' e decimale ','
res_df.round(3).to_csv(out_csv, index=False, sep=';', decimal=',')

print(f"\n{'='*60}")
print(f" Elaborazione Completata con Successo!")
print(f" File di uscita generato: {out_csv}")
print(f"{'='*60}")