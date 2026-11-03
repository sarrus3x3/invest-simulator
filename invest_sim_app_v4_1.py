"""
過去相場シミュレーター v4.1（Streamlit）
- 株価・為替データは、スクリプトと同じ場所の data フォルダに CSV で保存して使う。
- 更新は「追記のみ」。保存済みの過去データは上書きしない（更新前にバックアップも作成）。
- 過去データが取得元で変わっていた場合は、件数を警告で表示する。

タブ1: 早見表（開始年 × 保有年数のヒートマップ、暴落年を表示）
タブ2: シミュレーション（一括／積立）

起動:
    python -m pip install streamlit pandas yfinance
    python -m streamlit run invest_sim_app_v4_1.py

フォルダ構成:
    invest_sim_app_v4_1.py
    data/
        ACWI.csv ← アプリが yfinance から取得して保存（Close＋Dividends）
        USDJPY.csv ← アプリが yfinance から取得して保存
        FRED_DEXJPUS.csv ← アプリが FRED から取得して保存（1971〜）
        msci_acwi_net.csv ← MSCI公式サイトから手動ダウンロードして置く（名前は任意）
        backup/ ← 更新前の自動バックアップ
"""
from __future__ import annotations

import datetime as dt
import io
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="過去相場シミュレーター", page_icon="📈", layout="wide")

DATA_DIR = Path(__file__).resolve().parent / "data"
BACKUP_DIR = DATA_DIR / "backup"

DATE_COLS = ["Date", "date", "日付", "年月日", "データ日付", "Datetime", "DATE", "observation_date"]
CLOSE_COLS = ["Close", "close", "終値", "Price", "price", "DEXJPUS"]
ADJ_COLS = ["Adj Close", "Adj_Close", "adj_close", "調整後終値"]
DIV_COLS = ["Dividends", "dividends", "配当"]

CRASH_EVENTS = {
    1990: "湾岸危機・日本バブル崩壊",
    2000: "ITバブル崩壊（〜2002）",
    2001: "ITバブル崩壊・同時多発テロ",
    2002: "ITバブル崩壊の底",
    2008: "リーマン・ショック",
    2011: "欧州債務危機・超円高",
    2020: "コロナ・ショック",
    2022: "利上げ相場",
}

# ---------------------------------------------------------------- データ（読み込み）
def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", ""), errors="coerce")

def parse_csv_bytes(raw: bytes) -> pd.DataFrame:
    """CSV → DataFrame(index=日付, columns=['close','adj'])。attrs['kind'] に配当の扱いを記録。"""
    df = None
    for enc in ("utf-8-sig", "cp932"):
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc)
            break
        except Exception:
            continue
    if df is None:
        raise ValueError("CSVを読み込めませんでした")
    date_col = next((c for c in DATE_COLS if c in df.columns), df.columns[0])
    close_col = next((c for c in CLOSE_COLS if c in df.columns), None)
    adj_col = next((c for c in ADJ_COLS if c in df.columns), None)
    div_col = next((c for c in DIV_COLS if c in df.columns), None)
    if close_col is None and adj_col is None:
        if len(df.columns) < 2:
            raise ValueError(f"値の列が見つかりません。列名: {list(df.columns)}")
        close_col = df.columns[1] # MSCI等の独自列名は2列目を値とみなす

    idx = pd.to_datetime(df[date_col], errors="coerce")
    close = pd.Series(_num(df[close_col] if close_col else df[adj_col]).values, index=idx)
    out = pd.DataFrame({"close": close})
    if div_col is not None:
        out["div"] = _num(df[div_col]).fillna(0).values
    elif adj_col is not None:
        out["adj"] = _num(df[adj_col]).values
    out = out[~out.index.isna()].dropna(subset=["close"]).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    if out.empty:
        raise ValueError("有効なデータ行がありません")

    if "div" in out: # 終値＋配当 → 配当再投資の系列を自前で計算
        r = (out["close"] + out["div"]) / out["close"].shift(1)
        r.iloc[0] = 1.0
        out["adj"] = out["close"].iloc[0] * r.cumprod()
        out.attrs["kind"] = "dividends"
        out = out.drop(columns="div")
    elif "adj" in out:
        out["adj"] = out["adj"].fillna(out["close"])
        out.attrs["kind"] = "adj"
    else:
        out["adj"] = out["close"]
        out.attrs["kind"] = "plain"
    return out

@st.cache_data
def load_local(name: str, mtime: float) -> pd.DataFrame: # mtime はキャッシュ更新用
    return parse_csv_bytes((DATA_DIR / name).read_bytes())

def list_csvs() -> list[str]:
    DATA_DIR.mkdir(exist_ok=True)
    return sorted(p.name for p in DATA_DIR.glob("*.csv"))

# ---------------------------------------------------------------- データ（取得・追記保存）
def fetch_yf_raw(ticker: str, start: str = "1970-01-01") -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(ticker, start=start, auto_adjust=False, actions=True, progress=False)
    if df is None or df.empty:
        raise ValueError(f"{ticker} のデータを取得できませんでした")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    out = pd.DataFrame({"Close": df["Close"]})
    out["Dividends"] = df["Dividends"] if "Dividends" in df.columns else 0.0
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out.index.name = "Date"
    return out.dropna(subset=["Close"])

def fetch_fred_dexjpus() -> pd.DataFrame:
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXJPUS"
    df = pd.read_csv(url)
    date_col = next(c for c in df.columns if c.lower() in ("date", "observation_date"))
    out = pd.DataFrame({"Close": _num(df["DEXJPUS"]).values},
                       index=pd.to_datetime(df[date_col]))
    out.index.name = "Date"
    return out.dropna()

def merge_and_save(name: str, new: pd.DataFrame) -> tuple[int, int]:
    """追記のみで保存。戻り値: (追加行数, 過去データの食い違い件数)"""
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / name
    if not path.exists():
        new.to_csv(path)
        return len(new), 0
    BACKUP_DIR.mkdir(exist_ok=True)
    shutil.copy(path, BACKUP_DIR / f"{path.stem}_{dt.datetime.now():%Y%m%d_%H%M%S}.csv")
    old = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    common = old.index.intersection(new.index)
    diff = int((abs(new.loc[common, "Close"] / old.loc[common, "Close"] - 1) > 0.001).sum())
    added = new[new.index > old.index.max()]
    merged = pd.concat([old, added[old.columns.intersection(added.columns)]])
    merged.index.name = "Date"
    merged.to_csv(path)
    return len(added), diff

@st.cache_data
def make_dummy() -> tuple[pd.DataFrame, pd.DataFrame]:
    """動作確認用の人工データ（実際の相場ではない）"""
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("1995-01-02", dt.date.today())
    ret = rng.normal(0.0003, 0.011, len(idx))
    for s, e, d in [("2000-09-01", "2002-10-09", -0.0015), ("2008-09-15", "2009-03-09", -0.006),
                    ("2020-02-20", "2020-03-23", -0.012)]:
        ret[(idx >= s) & (idx <= e)] += d
    c = pd.Series(100 * np.exp(np.cumsum(ret)), index=idx)
    a = c * np.exp(np.arange(len(idx)) * 0.02 / 252)
    fx = pd.Series(110 * np.exp(np.cumsum(rng.normal(0, 0.004, len(idx)))), index=idx)
    return pd.DataFrame({"close": c, "adj": a}), pd.DataFrame({"close": fx, "adj": fx})

def build_series(px, fx, use_div, use_fx) -> pd.Series:
    s = px["adj" if use_div else "close"].copy()
    if use_fx:
        if fx is None:
            raise ValueError("円換算にはUSD/JPYのデータが必要です")
        r = fx["close"].reindex(s.index.union(fx.index)).ffill().reindex(s.index).bfill()
        s = s * r
    s.name = "price"
    return s

# ---------------------------------------------------------------- 計算
def simulate(p_all, start, years, method, lump, monthly) -> pd.DataFrame:
    end = start + pd.DateOffset(years=years)
    p = p_all[(p_all.index >= start) & (p_all.index < end)]
    if len(p) < 20:
        raise ValueError("選択した期間のデータが不足しています")
    out = pd.DataFrame(index=p.index)
    if method in ("一括", "比較"):
        out["一括_評価額"] = lump / p.iloc[0] * p
        out["一括_元本"] = lump
    if method in ("積立", "比較"):
        buy = p.groupby(p.index.to_period("M")).head(1).index
        u = pd.Series(0.0, index=p.index)
        u.loc[buy] = monthly / p.loc[buy]
        out["積立_評価額"] = u.cumsum() * p
        out["積立_元本"] = pd.Series(monthly, index=p.index).where(p.index.isin(buy), 0).cumsum()
    return out

def heatmap_table(p: pd.Series, max_years: int) -> pd.DataFrame:
    """開始年 × 保有年数の損益率と、元本割れ関連の指標（一括投資）"""
    rows = []
    last = p.index.max()
    for y in range(p.index.min().year, last.year + 1):
        sub = p[p.index >= pd.Timestamp(y, 1, 1)]
        if sub.empty or sub.index[0].year != y:
            continue
        t0, p0 = sub.index[0], sub.iloc[0]
        if t0 > pd.Timestamp(y, 1, 31): # 年初データがない年は除外
            continue
        rel = sub / p0
        row = {"開始年": y}
        for h in range(1, max_years + 1):
            th = t0 + pd.DateOffset(years=h)
            if th > last:
                row[h] = np.nan
            else:
                row[h] = rel[rel.index >= th].iloc[0] - 1
        # 元本割れ関連（データ末尾まで）
        under = rel < 1
        row["最大含み損"] = min(0.0, rel.min() - 1)
        months_under = rel[under].index.to_period("M").nunique()
        row["元本割れ月数"] = months_under
        if not under.any():
            row["元本回復"] = "割れず"
        else:
            last_under = rel[under].index.max()
            if last_under >= rel.index[-1]:
                row["元本回復"] = "未回復"
            else:
                yrs = (last_under - t0).days / 365.25
                row["元本回復"] = f"{yrs:.1f}年後"
        rows.append(row)
    return pd.DataFrame(rows).set_index("開始年")

def cell_color(v: float) -> str:
    if pd.isna(v):
        return "#f0f0f0"
    if v >= 0:
        a = min(v / 1.5, 1.0) # +150%で最も濃い緑
        return f"rgba(34,139,34,{0.15 + 0.75 * a:.2f})"
    a = min(-v / 0.5, 1.0) # -50%で最も濃い赤
    return f"rgba(220,20,60,{0.15 + 0.75 * a:.2f})"

def heatmap_frame(tbl: pd.DataFrame, years: list[int], mode: str, amount: float):
    """表示用の表（Styler）。HTMLタグを使わずに色付けする。"""
    labels = [f"{y} 📉{CRASH_EVENTS[int(y)]}" if int(y) in CRASH_EVENTS else str(y) for y in tbl.index]
    cols = years + ["最大含み損"]
    if mode == "損益率":
        body = tbl[cols].copy()
        fmt = "{:+.0%}"
    else:
        body = amount * (1 + tbl[cols]) / 10_000
        fmt = "{:,.0f}"
    view = body.copy()
    view.columns = [f"{c}年後" if isinstance(c, int) else c for c in cols]
    view["元本割れ月数"] = tbl["元本割れ月数"].astype(int)
    view["元本回復"] = tbl["元本回復"]
    view.index = labels
    view.index.name = "開始年"
    color_src = tbl[cols].copy()
    color_src.columns = view.columns[:len(cols)]
    color_src.index = labels

    def bg(_):
        return color_src.apply(lambda c: [f"background-color: {cell_color(v)}" for v in c])

    num_cols = list(view.columns[:len(cols)])
    return (view.style.apply(bg, axis=None, subset=num_cols)
            .format(fmt, subset=num_cols, na_rep=""))

def yen(x):
    return f"{x:,.0f} 円"

# ---------------------------------------------------------------- サイドバー
st.title("📈 過去相場シミュレーター")

with st.sidebar:
    st.header("データ（data フォルダ）")
    st.caption(f"保存先: {DATA_DIR}")

    with st.expander("データを取得・更新する", expanded=not list_csvs()):
        st.caption("更新は追記のみ。保存済みの過去データは書き換えません（更新前に backup に退避）。")
        tk = st.text_input("ティッカー（Yahoo）", "ACWI")
        if st.button(f"{tk} を yfinance から取得して保存"):
            try:
                n, d = merge_and_save(f"{tk.replace('^', '').replace('=', '')}.csv", fetch_yf_raw(tk))
                st.success(f"{n:,} 行を追加しました")
                if d:
                    st.warning(f"保存済みの過去データと取得元で {d:,} 件の値の食い違いがありました（保存側は変更していません）")
            except ImportError:
                st.error("yfinance 未インストールです")
            except Exception as e: # noqa: BLE001
                st.error(f"取得エラー: {e}")
        c1, c2 = st.columns(2)
        if c1.button("ドル円（Yahoo）"):
            try:
                n, d = merge_and_save("USDJPY.csv", fetch_yf_raw("JPY=X").drop(columns="Dividends"))
                st.success(f"USDJPY.csv に {n:,} 行追加")
                if d:
                    st.warning(f"過去データの食い違い {d:,} 件（保存側は変更なし）")
            except Exception as e: # noqa: BLE001
                st.error(f"取得エラー: {e}")
        if c2.button("ドル円（FRED 1971〜）"):
            try:
                n, d = merge_and_save("FRED_DEXJPUS.csv", fetch_fred_dexjpus())
                st.success(f"FRED_DEXJPUS.csv に {n:,} 行追加")
                if d:
                    st.warning(f"過去データの食い違い {d:,} 件（保存側は変更なし）")
            except Exception as e: # noqa: BLE001
                st.error(f"取得エラー: {e}")
        up = st.file_uploader("手動ダウンロードしたCSVを data に保存（MSCI等）", type=["csv"])
        if up is not None and st.button("このCSVを保存"):
            dest = DATA_DIR / up.name
            if dest.exists():
                st.error("同名のファイルがあります。ファイル名を変えてください（上書きはしません）")
            else:
                dest.write_bytes(up.getvalue())
                st.success(f"{up.name} を保存しました")

    use_dummy = st.checkbox("ダミーデータで試す（実際の相場ではありません）")
    files = list_csvs()
    use_fx = st.checkbox("円換算する（USD/JPY）", value=True,
                         help="オルカンは為替ヘッジなし。円高は評価額にマイナス、円安はプラス。")
    px = fx = None
    use_div = True
    ticker = "ダミー"
    if use_dummy:
        px, fx = make_dummy()
    elif files:
        def _default(cands):
            return next((files.index(c) for c in cands if c in files), 0)
        price_file = st.selectbox("価格・指数のファイル", files, index=_default(["ACWI.csv"]))
        ticker = Path(price_file).stem
        try:
            px = load_local(price_file, (DATA_DIR / price_file).stat().st_mtime)
            kind = px.attrs.get("kind")
            if kind == "dividends":
                use_div = st.checkbox("配当を含める（配当再投資で計算）", value=True)
            elif kind == "adj":
                use_div = st.checkbox("配当を含める（Adj Close を使用）", value=True)
            else:
                use_div = st.checkbox("このファイルは配当込みの指数（MSCI Net 等）", value=True,
                                      help="値そのものが配当込みならオン。オフなら配当なしとして表示。")
                px = px.assign(adj=px["close"])
            if use_fx:
                fx_file = st.selectbox("ドル円のファイル", files,
                                       index=_default(["FRED_DEXJPUS.csv", "USDJPY.csv"]))
                fx = load_local(fx_file, (DATA_DIR / fx_file).stat().st_mtime)
        except Exception as e: # noqa: BLE001
            st.error(f"データ読み込みエラー: {e}")
            px = None

    prices = None
    if px is not None and (not use_fx or fx is not None):
        try:
            prices = build_series(px, fx, use_div, use_fx)
            st.success(f"{prices.index.min():%Y-%m-%d} 〜 {prices.index.max():%Y-%m-%d}")
        except Exception as e: # noqa: BLE001
            st.error(str(e))

if prices is None:
    st.info("左のサイドバーでデータを用意してください。初回は「データを取得・更新する」から取得するか、"
            "「ダミーデータで試す」で画面を確認できます。")
    st.stop()

cond = f"{'配当込み' if use_div else '配当なし'}・{'円換算' if use_fx else 'ドル建て'}"
tab_hm, tab_sim = st.tabs(["🗺️ 早見表（ヒートマップ）", "📊 シミュレーション"])

# ---------------------------------------------------------------- 早見表
with tab_hm:
    st.subheader(f"早見表：その年の年初に一括投資したら、N年後どうなっていたか（{ticker}・{cond}）")
    c1, c2, c3 = st.columns(3)
    mode = c1.radio("表示", ["損益率", "評価額（万円）"], horizontal=True)
    amount = c2.number_input("投資額（円）", 100_000, 1_000_000_000, 30_000_000, step=1_000_000,
                             disabled=(mode == "損益率"))
    max_y = c3.slider("最大保有年数", 5, 30, 20)
    step = st.radio("表示する保有年数", ["毎年", "1・3・5・10・15・20…年"], horizontal=True)
    years = list(range(1, max_y + 1)) if step == "毎年" else         [y for y in [1, 2, 3, 5, 7, 10, 15, 20, 25, 30] if y <= max_y]

    tbl = heatmap_table(prices, max_y)
    if tbl.empty:
        st.error("年初から始まるデータがありません。")
    else:
        st.dataframe(heatmap_frame(tbl, years, mode, float(amount)), height=640)

        # まとめ
        st.markdown("#### まとめ")
        cols = st.columns(len([y for y in [1, 5, 10, 15] if y <= max_y]))
        for col, y in zip(cols, [y for y in [1, 5, 10, 15] if y <= max_y]):
            v = tbl[y].dropna()
            if len(v):
                col.metric(f"{y}年後に元本割れだった割合", f"{(v < 0).mean():.0%}",
                           help=f"{len(v)}ケース中 {(v < 0).sum()}ケース")
        worst = tbl["最大含み損"].idxmin()
        st.markdown(
            f"- 最大含み損がいちばん大きかったのは **{worst}年開始**："
            f"**{tbl.loc[worst, '最大含み損']:.0%}**（元本回復：{tbl.loc[worst, '元本回復']}、"
            f"元本割れ {int(tbl.loc[worst, '元本割れ月数'])}か月）\n"
            f"- 「元本割れ月数」×年間生活費÷12 が、オルカンを売らずに耐えるための生活防衛資金の目安です。"
        )
        st.caption("読み方：横に見る＝その年に始めた人の経過。縦に見る＝始めた時期によるばらつき。"
                   "暴落年より前の行が緑なら、含み益で暴落を吸収できていたことを示します。"
                   "📉は暴落・急落のあった年。元本割れ月数・元本回復はデータ末尾までの値。")

# ---------------------------------------------------------------- シミュレーション
with tab_sim:
    c1, c2, c3 = st.columns(3)
    start_date = c1.date_input("開始日", dt.date(2008, 9, 1),
                               min_value=prices.index.min().date(), max_value=prices.index.max().date())
    years_sim = c2.slider("運用期間（年）", 1, 30, 10)
    method = c3.radio("投資方法", ["一括", "積立", "比較"], horizontal=True)
    lump = monthly = 0.0
    if method == "一括":
        lump = st.number_input("一括投資額（円）", 10_000, 1_000_000_000, 3_000_000, step=100_000)
    else:
        monthly = st.number_input("毎月の積立額（円）", 1_000, 10_000_000, 30_000, step=1_000)
        if method == "比較":
            lump = monthly * years_sim * 12
            st.caption(f"一括は同額総投資: {yen(lump)}")
    try:
        sim = simulate(prices, pd.Timestamp(start_date), years_sim, method, float(lump), float(monthly))
    except ValueError as e:
        st.error(str(e))
        st.stop()
    targets = ["一括", "積立"] if method == "比較" else [method]
    for col, n in zip(st.columns(len(targets)), targets):
        v, pr = sim[f"{n}_評価額"], sim[f"{n}_元本"]
        pl = v.iloc[-1] - pr.iloc[-1]
        with col:
            st.metric(f"{n}：最終評価額", yen(v.iloc[-1]), f"{pl:+,.0f} 円（{pl / pr.iloc[-1]:+.1%}）")
            st.caption(f"最大下落率 {((v - v.cummax()) / v.cummax()).min():.1%} ／ "
                       f"元本割れ {(v < pr).sum():,}営業日")
    chart = pd.DataFrame(index=sim.index)
    for n in targets:
        chart[f"{n}（評価額）"] = sim[f"{n}_評価額"]
        chart[f"{n}（元本）"] = sim[f"{n}_元本"]
    st.line_chart(chart)

st.caption("※ 手数料・税金は考慮していません。ETF・指数で投資信託を近似したものです。"
           "過去の実績は将来の成果を保証するものではなく、投資の勧誘・助言ではありません。")
