import os
import time
import pandas as pd
import numpy as np

# Disabilita l'uso dei segnali se si è su Windows (evita crash legati a signal.alarm)
import platform
if platform.system() != 'Windows':
    import signal
else:
    signal = None

# Installazione guidata via codice (opzionale, ma utile per GitHub Actions)
# Se usi GitHub Actions, è meglio definirle nel file requirements.txt
try:
    from tvDatafeed import TvDatafeed, Interval
except ImportError:
    os.system('pip install git+https://github.com/rongardF/tvdatafeed.git -q')
    from tvDatafeed import TvDatafeed, Interval

try:
    import yfinance as yf
except ImportError:
    os.system('pip install yfinance -q')
    import yfinance as yf

# ============================================================
# CONFIGURAZIONE FILE INPUT / OUTPUT
# ============================================================
INPUT_FILE = "Tadingview_SPNQDW_MIBDAX_assets.txt"
OUTPUT_CSV = "Tadingview_SPNQDW_MIBDAX_assets_drawdown_stats.csv"

# ============================================================
# CONNESSIONE TRADINGVIEW
# ============================================================
tv = TvDatafeed()

# ============================================================
# STEP 1 & 2: Lettura file specifico ticker/exchange
# ============================================================
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Errore: Il file di input '{INPUT_FILE}' non è stato trovato nella directory corrente.")

print(f"Lettura file di input: {INPUT_FILE}")
with open(INPUT_FILE, 'r') as f:
    lines = [l.strip() for l in f.readlines() if l.strip()]

tickers_exchange = []
for line in lines:
    parts = line.split()
    if len(parts) >= 2:
        tickers_exchange.append((parts[0].upper(), parts[1].upper()))
    else:
        print(f"  [SKIP] Riga non valida: '{line}'")

print(f"Ticker caricati con successo: {len(tickers_exchange)}")

# ============================================================
# HELPER: scarica storico DAILY con timeout e retry
# ============================================================
def get_max_history_tv(symbol: str, exchange: str, bars_per_call=5000, max_retries=3, timeout_sec=60):
    prev_len = 0
    n_bars   = bars_per_call

    for attempt in range(max_retries + 1):
        try:
            if signal and hasattr(signal, 'SIGALRM'):
                def _timeout_handler(signum, frame):
                    raise TimeoutError(f"Timeout dopo {timeout_sec}s")
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(timeout_sec)

            df = tv.get_hist(
                symbol=symbol,
                exchange=exchange,
                interval=Interval.in_daily,
                n_bars=n_bars
            )

            if signal and hasattr(signal, 'SIGALRM'):
                signal.alarm(0)

            if df is None or df.empty:
                return pd.DataFrame()

            curr_len = len(df)
            if curr_len <= prev_len or curr_len < n_bars:
                break
            prev_len = curr_len
            n_bars  += bars_per_call

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

# ============================================================
# HELPER: scarica storico WEEKLY
# ============================================================
def get_weekly_history_tv(symbol: str, exchange: str, n_bars: int = 100, max_retries: int = 3, timeout_sec: int = 60):
    for attempt in range(max_retries + 1):
        try:
            if signal and hasattr(signal, 'SIGALRM'):
                def _timeout_handler(signum, frame):
                    raise TimeoutError(f"Timeout weekly dopo {timeout_sec}s")
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(timeout_sec)

            df = tv.get_hist(
                symbol=symbol,
                exchange=exchange,
                interval=Interval.in_weekly,
                n_bars=n_bars
            )

            if signal and hasattr(signal, 'SIGALRM'):
                signal.alarm(0)

            if df is None or df.empty:
                return pd.DataFrame()

            df_out = pd.DataFrame(index=pd.to_datetime(df.index).tz_localize(None))
            df_out['Open']  = df['open'].values
            df_out['High']  = df['high'].values
            df_out['Low']   = df['low'].values
            df_out['Close'] = df['close'].values
            return df_out.sort_index()

        except Exception as e:
            if attempt < max_retries:
                wait = 3 * (attempt + 1)
                time.sleep(wait)
            else:
                return pd.DataFrame()

    return pd.DataFrame()

# ============================================================
# HELPER: calcolo body% candela precedente
# ============================================================
BODY_THRESHOLD = 70.0

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

# ============================================================
# HELPER: calcolo POC
# ============================================================
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

# ============================================================
# HELPER: recupero Market Cap via yfinance
# ============================================================
def get_market_cap(symbol: str, exchange: str) -> float | None:
    try:
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
dist_results = []
anno_corrente = pd.Timestamp.now().year
n_total = len(tickers_exchange)

print(f"\nElaborazione di {n_total} ticker in corso...")

for i, (symbol, exchange) in enumerate(tickers_exchange, 1):
    print(f"[{i}/{n_total}] Elaborazione {symbol} ({exchange})...")
    
    if i > 1:
        time.sleep(1.5)

    try:
        df = get_max_history_tv(symbol, exchange)
        if df.empty:
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
            continue

        current_price = float(df['Price'].iloc[-1])
        poc1, poc2, poc3 = calculate_three_pocs(df)

        if poc1 is not None and poc2 is not None and poc3 is not None:
            sopra_3poc = "YES" if (current_price > poc1 and current_price > poc2 and current_price > poc3) else "NO"
        else:
            sopra_3poc = "N/A"

        tv_link = build_tv_link(exchange, symbol)
        ath_price = float(df['Price'].max())
        dd_da_ath_pct = round(((current_price - ath_price) / ath_price) * 100, 2)

        yd_curr = df[df.index.year == anno_corrente].copy()
        if len(yd_curr) >= 5:
            yd_curr['Peak']     = yd_curr['Price'].cummax()
            yd_curr['Drawdown'] = (yd_curr['Price'] - yd_curr['Peak']) / yd_curr['Peak']
            dd_corrente         = abs(yd_curr['Drawdown'].min() * 100)

            prezzo_inizio_anno = float(yd_curr['Price'].iloc[0])
            prezzo_ultimo      = float(yd_curr['Price'].iloc[-1])
            dd_ytd_pct         = round(((prezzo_ultimo - prezzo_inizio_anno) / prezzo_inizio_anno) * 100, 2)

            hist_mdd = [v for v, yr in zip(yearly_mdd, [y for y in years if len(df[df.index.year==y])>=20]) if yr != anno_corrente]

            if hist_mdd:
                dd_med_st  = np.mean(hist_mdd)
                dd_max_st  = np.max(hist_mdd)
                std_dev    = np.std(hist_mdd)
                distanza   = dd_corrente - dd_med_st
                cv         = (std_dev / dd_med_st * 100) if dd_med_st > 0 else 0
                affid      = max(0, min(100, 100 - cv))

                # Generazione del dataset completo a 22 colonne
                dist_results.append({
                    'Ticker':               symbol,
                    'Exchange':             exchange,
                    'Prezzo':               round(current_price, 2),
                    'POC1':                 poc1,
                    'POC2':                 poc2,
                    'POC3':                 poc3,
                    'Sopra_3POC':           sopra_3poc,
                    'DD_da_ATH%':           dd_da_ath_pct,
                    'DD_YTD%':              dd_ytd_pct,
                    f'DD_{anno_corrente}%': round(dd_corrente, 1),
                    'DD_Medio_Storico%':    round(dd_med_st, 1),
                    'Max_DD_Storico%':      round(dd_max_st, 1),
                    'Distanza%':            round(distanza, 1),
                    'StdDev%':              round(std_dev, 1),
                    'CV%':                  round(cv, 1),
                    'Affidabilità%':        round(affid, 1),
                    'Body_D%':              body_d_pct,
                    'Body_D_Flag':          body_d_flag,
                    'Body_W%':              body_w_pct,
                    'Body_W_Flag':          body_w_flag,
                    'Link_TV_W':            tv_link,
                    'Market_Cap':           market_cap
                })

    except Exception as e:
        print(f"Errore sul ticker {symbol}: {e}")

# ============================================================
# SALVATAGGIO CSV (Esattamente le 22 colonne desiderate)
# ============================================================
if dist_results:
    dist_df = pd.DataFrame(dist_results)
    
    # Ordinamento opzionale coerente con la logica precedente (Sopra_3POC e performance)
    if 'Sopra_3POC' in dist_df.columns:
        dist_df['_sort_flag'] = dist_df['Sopra_3POC'].map({'YES': 0, 'NO': 1, 'N/A': 2})
        dist_df = dist_df.sort_values(['_sort_flag', 'Ticker'], ascending=[True, True]).drop(columns='_sort_flag')

    # Salvataggio finale
    dist_df.to_csv(OUTPUT_CSV, index=False, sep=';', decimal=',')
    print(f"\n[SUCCESS] File generato correttamente: {OUTPUT_CSV}")
    print(f"Numero totale di colonne: {len(dist_df.columns)} (Dovrebbero essere 22)")
    print(f"Righe elaborate: {len(dist_df)}")
else:
    print("\n[ERROR] Nessun dato estratto. Verifica la connessione o l'integrità del file TXT.")
