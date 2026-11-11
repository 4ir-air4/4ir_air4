"""NTSB 미국 사고 DB와 국토교통부 사고 자료의 START_YEAR 이후 통계, 사고의 사망 발생 여부 사후 분석."""
import warnings
import zipfile

import pandas as pd

from ldh.core.data import CACHE, REF, START_YEAR, read_csv_any
from ldh.core.ml import binary_result, encode, model_row, self_check

KOREA = REF / "korea"
LIGHT = {"DAYL": "주간", "NITE": "야간", "NBRT": "야간(밝음)", "NDRK": "야간(어두움)", "DAWN": "새벽", "DUSK": "황혼"}
WX_BASIC = {"VMC": "시계 비행 기상", "IMC": "계기 비행 기상", "UNK": "미상"}
CATEGORY = {"AIR": "비행기", "HELI": "헬리콥터", "GLI": "글라이더", "BALL": "기구", "WSFT": "체중 이동", "PLFT": "파워 리프트", "GYRO": "자이로", "ULTR": "초경량", "BLIM": "비행선", "PPAR": "동력 낙하산"}
SQL = """select e.ev_year, e.ev_month, e.ev_time, e.ev_state, e.light_cond, e.wx_cond_basic, e.wind_vel_kts, e.vis_sm,
                e.ev_highest_injury, e.inj_tot_f, a.far_part, a.acft_category, a.num_eng, a.homebuilt, a.damage
         from events as e inner join aircraft as a on e.ev_id = a.ev_id
         where e.ev_type = 'ACC' and e.ev_country = 'USA' and a.Aircraft_Key = 1"""
TEXT = ["ev_state", "light_cond", "wx_cond_basic", "ev_highest_injury", "far_part", "acft_category", "homebuilt", "damage"]


def read_ntsb(name):
    """압축을 풀어 필요한 열만 parquet 으로 저장한 뒤 큰 mdb 는 지운다."""
    path = CACHE / f"ntsb_{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    import pyodbc
    CACHE.mkdir(parents=True, exist_ok=True)
    zip_path = next((REF / "ntsb").glob(f"{name}.zip"))
    with zipfile.ZipFile(zip_path) as z:
        member = z.namelist()[0]
        z.extract(member, CACHE)
    mdb = CACHE / member
    try:
        con = pyodbc.connect(f"Driver={{Microsoft Access Driver (*.mdb, *.accdb)}};Dbq={mdb.resolve()};")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            d = pd.read_sql(SQL, con)
        con.close()
    finally:
        mdb.unlink()
    for c in TEXT:
        d[c] = d[c].astype("string").str.strip()
    d.to_parquet(path, index=False)
    return d


def load_events():
    # shortcut: avall 은 2008년부터라 START_YEAR 를 2008 미만으로 낮추면 read_ntsb("Pre2008") 도 합쳐야 한다
    d = read_ntsb("avall")
    return d[d.ev_year >= START_YEAR].reset_index(drop=True)


def since_start(d):
    """일자 열이 START_YEAR 이후인 행만 남긴다."""
    return d[pd.to_datetime(d["일자"], errors="coerce").dt.year >= START_YEAR]


def run(store):
    self_check()
    d = load_events()
    d["사망 발생"] = d.ev_highest_injury.eq("FATL").fillna(False)
    d["hour"] = pd.to_numeric(d.ev_time, errors="coerce") // 100
    g = d.groupby("ev_year")
    yearly = pd.DataFrame({"사고": g.size(), "사망 사고": g["사망 발생"].sum(), "사망자": g.inj_tot_f.sum()}).reset_index().rename(columns={"ev_year": "연도"})
    yearly["사망 사고 비율"] = yearly["사망 사고"] / yearly.사고
    store.save_table("yearly", yearly)

    def by(col, names, label):
        t = d.groupby(col).agg(사고=("사망 발생", "size"), 사망사고=("사망 발생", "sum")).reset_index()
        t[label] = t[col].map(names).fillna(t[col]) if names else t[col]
        t["사망 사고 비율"] = t.사망사고 / t.사고
        return t[t.사고 >= 30][[label, "사고", "사망 사고 비율"]].sort_values("사고", ascending=False)

    store.save_table("light", by("light_cond", LIGHT, "조명"))
    store.save_table("wx", by("wx_cond_basic", WX_BASIC, "기상"))
    store.save_table("category", by("acft_category", CATEGORY, "항공기 종류"))
    store.save_table("far", by("far_part", {}, "운항 규정(FAR)").head(10))
    month = d.groupby("ev_month").size().rename("사고").reset_index().rename(columns={"ev_month": "월"})
    store.save_table("month", month)

    # 손상 정도·부상자 수는 사고 결과이므로 특성에서 뺀다
    m = d.copy()
    m["y"] = m["사망 발생"].astype(int)
    tr, val, te = m[m.ev_year <= 2022], m[m.ev_year == 2023], m[m.ev_year >= 2024]
    cat = ["ev_state", "light_cond", "wx_cond_basic", "far_part", "acft_category", "homebuilt"]
    X_tr, X_val, X_te = encode(tr, [val, te], cat, ["hour", "ev_month", "wind_vel_kts", "vis_sm", "num_eng"])
    models = [model_row("NTSB 사고의 사망 발생 여부 (사후 분석)", binary_result(X_tr, tr.y, X_val, val.y, X_te, te.y, class_weight="balanced"),
                        f"{START_YEAR}~2022", "2024~2026-10", "사고 환경, 기체 정보", "사고가 난 뒤의 분류이며 사고 발생 예측이 아닙니다")]

    dom = read_csv_any(KOREA / "국토교통부_연도별_국내주요항공사고.csv").rename(columns={"구분": "연도"})
    dom = dom[pd.to_numeric(dom.연도, errors="coerce").notna()].assign(연도=lambda x: x.연도.astype(int))
    dom = dom[dom.연도 >= START_YEAR]
    store.save_table("korea_yearly", dom[["연도", "사고발생건수", "준사고발생건수", "경량사고발생건수", "초경량사고발생건수", "총발생건수", "총사망수"]])
    cases = since_start(read_csv_any(KOREA / "국토교통부_사고유형별_국내항공사고.csv"))
    store.save_table("korea_phase", cases["비행단계"].value_counts().rename("건수").reset_index().head(10))
    store.save_table("korea_type", cases["사고구분"].value_counts().rename("건수").reset_index())
    world = since_start(read_csv_any(KOREA / "국토교통부_연도별_전세계항공사고.csv"))
    store.save_table("world_phase", world["비행단계"].value_counts().rename("건수").reset_index().head(10))

    last = yearly[yearly.연도 == 2025].iloc[0]
    first = yearly[yearly.연도 == START_YEAR].iloc[0]
    store.save_summary(
        headline=f"미국 항공 사고는 {START_YEAR}년 {first.사고:,.0f}건, 2025년 {last.사고:,.0f}건이고, 2025년 사고 중 사망 사고 비율은 약 {last['사망 사고 비율']:.0%}입니다. 사고 환경 정보만으로는 사망 여부를 잘 가려내지 못합니다.",
        kpis=[("분석한 미국 사고", f"{len(d):,}건", f"{START_YEAR}~2026-10"), ("2025년 사고", f"{last.사고:,.0f}건"),
              ("2025년 사망 사고", f"{last['사망 사고']:,.0f}건"), ("2025년 사망자", f"{last.사망자:,.0f}명")],
        models=models,
        notes=[f"NTSB 기준 미국 내 사고(ACC)만 집계했습니다. 준사고(INC)와 {START_YEAR}년 이전 자료는 제외했습니다.",
               "2020년대 이후 일부 건은 조사 중이라 값이 나중에 바뀔 수 있습니다.",
               f"국내 사고 건별 기록은 {START_YEAR}년 이후 {len(cases)}건으로 적어 통계만 제공하고 모델은 만들지 않았습니다.",
               "상관 통계입니다. 비율 차이는 기종·운항 유형 차이도 함께 반영합니다."],
        sources=["NTSB 사고 DB (avall)", "국토교통부 항공사고 자료"],
    )
