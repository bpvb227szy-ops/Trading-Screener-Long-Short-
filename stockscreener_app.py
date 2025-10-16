import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import requests
from io import StringIO
import time, random

st.set_page_config(page_title="Stock Screener + Blog", layout="wide")

# ===================== UI: Seitenwahl =====================
pg = st.sidebar.radio("Seite wählen", ["🧠 Screener", "✍️ Blog"], index=0)

# ===================== Shared Settings =====================
PERIOD = "1y"          # Historie
INTERVAL = "1d"        # Tagesdaten
RSI_LEN = 14
VOL_SMA_WIN = 20
BATCH_SIZE = 60

# ===================== Utils =====================
def read_html_with_headers(url: str):
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return pd.read_html(StringIO(r.text))

@st.cache_data(show_spinner=True)
def load_index_constituents(flags: dict) -> pd.DataFrame:
    """
    flags: {"sp500":bool,"nasdaq100":bool,"dax40":bool,"kospi200":bool,"nikkei225":bool}
    Return: DF mit Spalten ['Ticker','Name','Sector'] (Sector optional)
    """
    parts = []

    # S&P 500
    if flags.get("sp500"):
        try:
            tables = read_html_with_headers("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
            sp = tables[0]
            part = sp.rename(columns={"Symbol":"Ticker","Security":"Name","GICS Sector":"Sector"})[["Ticker","Name","Sector"]]
            part["Ticker"] = part["Ticker"].astype(str).str.replace(".", "-", regex=False)
            parts.append(part)
        except Exception:
            parts.append(pd.DataFrame({
                "Ticker":["AAPL","MSFT","NVDA","AMZN","GOOGL","META","BRK-B","XOM","LLY","JPM"],
                "Name":["Apple","Microsoft","NVIDIA","Amazon","Alphabet","Meta","Berkshire","Exxon","Eli Lilly","JPMorgan"],
                "Sector":["Info Tech","Info Tech","Info Tech","Cons Disc","Comm","Comm","Financials","Energy","Health Care","Financials"]
            }))

    # NASDAQ-100
    if flags.get("nasdaq100"):
        try:
            tables = read_html_with_headers("https://en.wikipedia.org/wiki/Nasdaq-100")
            nq = None
            for t in tables:
                if "Ticker" in t.columns:
                    nq = t
                    break
            if nq is None:
                raise RuntimeError("No Ticker column for NASDAQ-100")
            part = nq.rename(columns={"Ticker":"Ticker","Company":"Name"})
            cols = [c for c in ["Ticker","Name","Sector"] if c in part.columns]
            part = part[cols]
            part["Ticker"] = part["Ticker"].astype(str).str.replace(".", "-", regex=False)
            parts.append(part)
        except Exception:
            parts.append(pd.DataFrame({
                "Ticker":["AAPL","MSFT","NVDA","AMZN","META","GOOGL","AVGO","PEP","COST","ADBE"],
                "Name":["Apple","Microsoft","NVIDIA","Amazon","Meta","Alphabet","Broadcom","PepsiCo","Costco","Adobe"]
            }))

    # DAX 40 (vereinfachte Heuristik für Ticker)
    if flags.get("dax40"):
        try:
            tables = read_html_with_headers("https://en.wikipedia.org/wiki/DAX")
            dax = tables[1]
            name_col = "Company" if "Company" in dax.columns else dax.columns[0]
            part = pd.DataFrame({"Name": dax[name_col].astype(str)})
            heur = part["Name"].str.upper().str.replace(r"[^A-Z0-9]", "", regex=True).str[:5]
            part["Ticker"] = heur + ".DE"
            parts.append(part[["Ticker","Name"]])
        except Exception:
            parts.append(pd.DataFrame({
                "Ticker":["SIE.DE","BMW.DE","VOW3.DE","BAS.DE","SAP.DE","ALV.DE","DTE.DE","RWE.DE","BAYN.DE","MUV2.DE"],
                "Name":["Siemens","BMW","Volkswagen","BASF","SAP","Allianz","Deutsche Telekom","RWE","Bayer","Munich Re"]
            }))

    # KOSPI 200 (Korea)
    if flags.get("kospi200"):
        try:
            tables = read_html_with_headers("https://en.wikipedia.org/wiki/KOSPI_200")
            k = tables[0]
            code_col = None
            for c in k.columns:
                if str(c).lower() in ["code","ticker","symbol","company code"]:
                    code_col = c
                    break
            if code_col is None:
                code_col = k.columns[0]
            codes = k[code_col].astype(str).str.zfill(6)
            part = pd.DataFrame({"Ticker": codes + ".KS"})
            parts.append(part)
        except Exception:
            parts.append(pd.DataFrame({
                "Ticker":["005930.KS","000660.KS","207940.KS","005380.KS","035420.KS"]
            }))

    # Nikkei 225 (Japan) – TSE: Suffix .T
    if flags.get("nikkei225"):
        try:
            tables = read_html_with_headers("https://en.wikipedia.org/wiki/Nikkei_225")
            nk = None
            for t in tables:
                cols = [c.lower() for c in t.columns.map(str)]
                if any("code" in c for c in cols) or any("symbol" in c for c in cols):
                    nk = t
                    break
            if nk is None:
                raise RuntimeError("No code column for Nikkei 225")
            # finde Code-Spalte
            code_col = None
            for c in nk.columns:
                if str(c).lower() in ["code","symbol","ticker"]:
                    code_col = c
                    break
            if code_col is None:
                code_col = nk.columns[0]
            codes = nk[code_col].astype(str).str.replace(r"[^0-9]", "", regex=True).str.zfill(4)
            # Nikkei hat teils 4-stellige Codes, TSE YF braucht meist 4-stellig + ".T"
            part = pd.DataFrame({"Ticker": codes + ".T"})
            parts.append(part)
        except Exception:
            parts.append(pd.DataFrame({
                "Ticker":["7203.T","6758.T","9984.T","9432.T","8035.T"]  # Toyota, Sony, SoftBank, NTT, Renesas/SCREEN vary – repräsentativ
            }))

    if not parts:
        return pd.DataFrame(columns=["Ticker","Name","Sector"])

    df = pd.concat(parts, ignore_index=True)
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    df = df.drop_duplicates(subset=["Ticker"]).reset_index(drop=True)
    return df[["Ticker"] + [c for c in ["Name","Sector"] if c in df.columns]]

@st.cache_data(show_spinner=True)
def download_in_batches(tickers, period="1y", interval="1d", batch_size=60, retries=2, pause=1.0):
    frames = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        ok = False
        for r in range(retries):
            try:
                df = yf.download(
                    batch, period=period, interval=interval,
                    auto_adjust=True, threads=True, group_by="ticker", progress=False
                )
                if df is not None and not df.empty:
                    frames.append(df)
                    ok = True
                    break
            except Exception:
                pass
            time.sleep(pause + random.random()*0.7)
        time.sleep(pause)
    if not frames:
        return None
    big = pd.concat(frames, axis=1).sort_index()
    return big

def get_df_for_ticker(data, ticker):
    try:
        if isinstance(data.columns, pd.MultiIndex):
            df = pd.DataFrame({
                "Close": data["Close"][ticker],
                "Volume": data["Volume"][ticker],
            }).dropna()
        else:
            df = data[["Close","Volume"]].dropna()
        return df
    except Exception:
        return None

# ===================== Indikatoren & Scoring =====================
def compute_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    avg_gain = up.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    avg_loss = down.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def compute_macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def compute_all_indicators(df: pd.DataFrame):
    close = df["Close"].dropna()
    vol = df["Volume"].dropna() if "Volume" in df.columns else pd.Series(dtype=float, index=close.index)
    rsi = compute_rsi(close, RSI_LEN)
    macd, macd_sig, macd_hist = compute_macd(close)
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    vol_sma = vol.rolling(VOL_SMA_WIN).mean()
    ind = {
        "last_price": float(close.iloc[-1]),
        "rsi": float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else np.nan,
        "ma50": float(ma50.iloc[-1]) if not np.isnan(ma50.iloc[-1]) else np.nan,
        "ma200": float(ma200.iloc[-1]) if not np.isnan(ma200.iloc[-1]) else np.nan,
        "macd": float(macd.iloc[-1]),
        "macd_sig": float(macd_sig.iloc[-1]),
        "macd_hist": float(macd_hist.iloc[-1]),
        "macd_prev": float(macd.iloc[-2]) if len(macd) > 1 else np.nan,
        "macd_sig_prev": float(macd_sig.iloc[-2]) if len(macd_sig) > 1 else np.nan,
        "volume": float(vol.iloc[-1]) if len(vol) else np.nan,
        "vol_sma": float(vol_sma.iloc[-1]) if len(vol_sma) else np.nan,
    }
    ind["vol_ratio"] = float(ind["volume"]/ind["vol_sma"]) if ind["vol_sma"] and ind["vol_sma"] > 0 else np.nan
    return ind

def macd_component(macd, sig, macd_prev, sig_prev, bullish: bool) -> float:
    spread = macd - sig
    base = float(np.tanh(abs(spread) * 3))
    good_now = (spread > 0) if bullish else (spread < 0)
    crossed_now = False
    if not (np.isnan(macd_prev) or np.isnan(sig_prev)):
        prev_spread = macd_prev - sig_prev
        crossed_now = (prev_spread <= 0 and spread > 0) if bullish else (prev_spread >= 0 and spread < 0)
    score = base
    if good_now: score = max(score, 0.6)
    if crossed_now: score = max(score, 0.9)
    return float(np.clip(score, 0, 1))

def rsi_component(rsi: float, bullish: bool) -> float:
    if np.isnan(rsi): return 0.0
    return float(np.clip((100 - rsi)/100.0 if bullish else rsi/100.0, 0, 1))

def vol_component(vr: float) -> float:
    if np.isnan(vr): return 0.0
    return float(np.clip(vr/2.0, 0, 1))  # 2x Durchschnitt = 1.0

def trend_component(price: float, ma200: float, bullish: bool) -> float:
    if np.isnan(ma200) or ma200 == 0: return 0.0
    prem = (price/ma200) - 1.0
    x = prem if bullish else -prem
    return float(np.clip(x/0.10, 0, 1))  # 10% über/unter 200d ~ 1.0

def score_signal(ind, bullish: bool) -> float:
    rsi_s = rsi_component(ind["rsi"], bullish)
    macd_s = macd_component(ind["macd"], ind["macd_sig"], ind["macd_prev"], ind["macd_sig_prev"], bullish)
    vol_s = vol_component(ind["vol_ratio"])
    trn_s = trend_component(ind["last_price"], ind["ma200"], bullish)
    return float(round((rsi_s*0.40 + macd_s*0.30 + vol_s*0.20 + trn_s*0.10)*100, 1))

# ===================== Seite: Screener =====================
if pg == "🧠 Screener":
    st.title("🧠 Aktien-Screener – Long/Short (RSI + MACD + Volumen)")

    st.sidebar.header("Universum & Filter")
    use_sp500 = st.sidebar.checkbox("S&P 500 (USA)", True)
    use_nasdaq100 = st.sidebar.checkbox("NASDAQ-100 (USA)", True)
    use_dax40 = st.sidebar.checkbox("DAX 40 (DE)", True)
    use_kospi200 = st.sidebar.checkbox("KOSPI 200 (KR)", False)
    use_nikkei225 = st.sidebar.checkbox("Nikkei 225 (JP)", True)

    min_price = st.sidebar.number_input("Mindestpreis (keine Pennystocks)", value=5.0, step=0.5)
    min_avg_volume = st.sidebar.number_input("Mindest-Ø-Volumen (20 Tage, Stück)", value=200000, step=50000)
    enforce_sector_mix = st.sidebar.checkbox("Sektor-Diversifikation (max 3 je Sektor in den Top-Listen)", True)
    max_per_sector = st.sidebar.slider("Max je Sektor (Top-Listen)", 1, 5, 3)

    uni = load_index_constituents({
        "sp500": use_sp500, "nasdaq100": use_nasdaq100,
        "dax40": use_dax40, "kospi200": use_kospi200, "nikkei225": use_nikkei225
    })
    ALL_TICKERS = uni["Ticker"].tolist()
    st.caption(f"Geladene Ticker gesamt: **{len(ALL_TICKERS)}**")

    if len(ALL_TICKERS) == 0:
        st.warning("Keine Ticker geladen – bitte mindestens einen Index aktivieren.")
        st.stop()

    data = download_in_batches(ALL_TICKERS, PERIOD, INTERVAL, batch_size=BATCH_SIZE)
    if data is None or data.empty:
        st.error("Konnte keine Marktdaten laden. Reduziere Universum (z. B. KOSPI aus) und klicke oben rechts auf Rerun.")
        st.stop()

    def sector_of(ticker):
        row = uni.loc[uni["Ticker"] == ticker]
        if not row.empty and "Sector" in row.columns:
            sec = row.iloc[0].get("Sector", np.nan)
            return sec if (isinstance(sec, str) and len(sec) > 0) else "Unknown"
        return "Unknown"

    long_rows, short_rows = [], []
    for tk in ALL_TICKERS:
        df_t = get_df_for_ticker(data, tk)
        if df_t is None or len(df_t) < 200:
            continue
        # Penny-/Illiquid-Filter
        if df_t["Close"].tail(20).mean() < min_price: continue
        if df_t["Volume"].tail(20).mean() < min_avg_volume: continue

        ind = compute_all_indicators(df_t)
        in_up = ind["last_price"] > ind["ma200"]
        in_down = ind["last_price"] < ind["ma200"]
        vol_ok = (not np.isnan(ind["vol_ratio"])) and (ind["vol_ratio"] >= 1.0)
        sec = sector_of(tk)

        if in_up and vol_ok:
            score = score_signal(ind, bullish=True)
            long_rows.append({
                "Ticker": tk, "Sector": sec, "Kurs": round(ind["last_price"],2),
                "RSI(14)": round(ind["rsi"],1) if not np.isnan(ind["rsi"]) else None,
                "MACD": round(ind["macd"],4), "MACD_Signal": round(ind["macd_sig"],4),
                "MACD_Hist": round(ind["macd_hist"],4),
                "Vol-Ratio(akt/20d)": round(ind["vol_ratio"],2) if not np.isnan(ind["vol_ratio"]) else None,
                "Long-Wahrscheinlichkeit (%)": score
            })
        if in_down and vol_ok:
            score = score_signal(ind, bullish=False)
            short_rows.append({
                "Ticker": tk, "Sector": sec, "Kurs": round(ind["last_price"],2),
                "RSI(14)": round(ind["rsi"],1) if not np.isnan(ind["rsi"]) else None,
                "MACD": round(ind["macd"],4), "MACD_Signal": round(ind["macd_sig"],4),
                "MACD_Hist": round(ind["macd_hist"],4),
                "Vol-Ratio(akt/20d)": round(ind["vol_ratio"],2) if not np.isnan(ind["vol_ratio"]) else None,
                "Short-Wahrscheinlichkeit (%)": score
            })

    def top_k_with_sector_cap(df, score_col, k=10, cap=3):
        if df.empty or not enforce_sector_mix:
            return df.sort_values(score_col, ascending=False).head(k)
        out, per_sec = [], {}
        for _, row in df.sort_values(score_col, ascending=False).iterrows():
            sec = row.get("Sector", "Unknown")
            per_sec.setdefault(sec, 0)
            if per_sec[sec] < cap:
                out.append(row)
                per_sec[sec] += 1
            if len(out) >= k: break
        return pd.DataFrame(out)

    df_long = pd.DataFrame(long_rows)
    df_short = pd.DataFrame(short_rows)
    df_long_top = top_k_with_sector_cap(df_long, "Long-Wahrscheinlichkeit (%)", 10, max_per_sector) if not df_long.empty else df_long
    df_short_top = top_k_with_sector_cap(df_short, "Short-Wahrscheinlichkeit (%)", 10, max_per_sector) if not df_short.empty else df_short

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Top 10 Long-Kandidaten")
        if not df_long_top.empty:
            st.dataframe(df_long_top.set_index("Ticker"), height=480)
        else:
            st.info("Keine Long-Kandidaten (Filter lockern?)")
    with c2:
        st.subheader("Top 10 Short-Kandidaten")
        if not df_short_top.empty:
            st.dataframe(df_short_top.set_index("Ticker"), height=480)
        else:
            st.info("Keine Short-Kandidaten (Filter lockern?)")

    st.markdown("---")
    st.subheader("🔎 Einzelsuche")
    search = st.text_input("Ticker (z. B. AAPL, MSFT, SIE.DE, 7203.T, 005930.KS):", "")
    if search:
        df_s = yf.download(search.strip(), period=PERIOD, interval=INTERVAL, auto_adjust=True)
        if df_s is not None and not df_s.empty:
            if "Volume" not in df_s.columns: df_s["Volume"] = np.nan
            ind = compute_all_indicators(df_s[["Close","Volume"]].dropna())
            trend = "Aufwärtstrend 🟢" if ind["last_price"] > ind["ma200"] else ("Abwärtstrend 🔴" if ind["last_price"] < ind["ma200"] else "Trend unklar")
            long_score = score_signal(ind, bullish=True)
            short_score = score_signal(ind, bullish=False)
            st.write(
                f"**{search.upper()}** – Kurs: **{ind['last_price']:.2f}**, 200d-SMA: **{ind['ma200']:.2f}**, "
                f"RSI(14): **{ind['rsi']:.1f}**, MACD: **{ind['macd']:.4f}**, Signal: **{ind['macd_sig']:.4f}**, "
                f"Vol-Ratio(akt/20d): **{(ind['vol_ratio'] if not np.isnan(ind['vol_ratio']) else 0):.2f}** → {trend}"
            )
            if ind["last_price"] > ind["ma200"]:
                st.success(f"Long-Score: **{long_score:.1f}%**  •  Short-Score: {short_score:.1f}%")
            elif ind["last_price"] < ind["ma200"]:
                st.error(f"Short-Score: **{short_score:.1f}%**  •  Long-Score: {long_score:.1f}%")
            else:
                st.info(f"Long-Score: {long_score:.1f}%  •  Short-Score: {short_score:.1f}% (Trend unklar)")
        else:
            st.warning("Keine Daten zum Ticker gefunden.")

# ===================== Seite: Blog =====================
if pg == "✍️ Blog":
    st.title("✍️ Dein Blog")
    st.caption("Einfach hier schreiben – exportiere deinen Text als Markdown-Datei oder lade vorhandene .md/.txt hoch.")

    st.subheader("Neuen Beitrag verfassen")
    title = st.text_input("Titel", "")
    body = st.text_area("Inhalt (Markdown möglich)", height=300, placeholder="Schreib hier deinen Post...")

    colA, colB, colC = st.columns([1,1,2])
    with colA:
        if st.button("⬇️ Als Markdown exportieren"):
            if len(title.strip()) == 0: st.warning("Bitte einen Titel eingeben."); st.stop()
            md = f"# {title.strip()}\n\n{body if body else ''}\n"
            st.download_button("Download .md", md, file_name=f"{title.strip().replace(' ','_')}.md")
    with colB:
        uploaded = st.file_uploader("Vorhandene .md/.txt laden", type=["md","txt"])
        if uploaded is not None:
            content = uploaded.read().decode("utf-8", errors="ignore")
            st.text_area("Datei-Inhalt", value=content, height=250)

    st.markdown("---")
    st.subheader("Schnell-Format")
    st.markdown("""
- **Fett**: `**Text**`
- *Kursiv*: `*Text*`
- Code: ```python  # dreifache Backticks
print("Hello")
