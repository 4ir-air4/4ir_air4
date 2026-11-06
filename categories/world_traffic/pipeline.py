"""OpenSky 항공편 목록(START_YEAR~2022), 세계은행 국가 통계, 공항·항공기 참조 자료의 통계와 비행 시간 예측.

OpenSky 조건에 따라 개별 항공기 식별자는 저장하지 않고 집계값만 남긴다.
"""
import io
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from core.data import REF, START_YEAR
from core.ml import encode, model_row, regression_result

SAMPLE_PER_MONTH = 50_000
WB = {"IS.AIR.PSGR": "여객(명)", "IS.AIR.DPRT": "운항(편)", "IS.AIR.GOOD.MT.K1": "화물(백만 톤-km)"}
WB_COUNTRIES = {"WLD": "세계", "USA": "미국", "CHN": "중국", "KOR": "한국", "JPN": "일본", "DEU": "독일"}
AIRPORT_TYPES = {"large_airport": "대형", "medium_airport": "중형", "small_airport": "소형", "heliport": "헬기장", "seaplane_base": "수상기", "closed": "폐쇄", "balloonport": "기구장"}


def airports():
    return pd.read_csv(REF / "ourairports/airports.csv", usecols=["ident", "latitude_deg", "longitude_deg", "iso_country", "type", "scheduled_service"], keep_default_na=False, na_values=[""])


def opensky_paths():
    return [p for p in sorted((REF / "opensky_flightlist").glob("flightlist_*.csv.gz")) if int(p.name.split("_")[1][:4]) >= START_YEAR]


def read_month(path):
    """한 달 파일을 집계로 줄인다. 전 기간 원본을 메모리에 올리지 않기 위함."""
    d = pd.read_csv(path, usecols=["origin", "destination", "typecode", "firstseen", "lastseen"], dtype="string")
    month = path.name.split("_")[1][:6]
    country = airports().set_index("ident").iso_country
    known = d.dropna(subset=["origin", "destination"])
    out = {"month": month, "total": len(d), "known": len(known),
           "country": known.origin.map(country).value_counts().rename("n").rename_axis("country").reset_index().assign(month=month),
           "type": d.typecode.value_counts().head(300).rename("n").rename_axis("typecode").reset_index().assign(year=month[:4]),
           "route": known.groupby(["origin", "destination"]).size().nlargest(3000).rename("n").reset_index().assign(year=month[:4])}
    first, last = pd.to_datetime(known.firstseen, utc=True), pd.to_datetime(known.lastseen, utc=True)
    s = known.assign(y=(last - first).dt.total_seconds() / 60, hour=first.dt.hour, dow=first.dt.day_of_week, period=month)
    s = s[s.y.between(5, 1200)]
    out["sample"] = s.sample(n=min(SAMPLE_PER_MONTH, len(s)), random_state=0)[["origin", "destination", "typecode", "y", "hour", "dow", "period"]]
    return out


def duration_model(sample):
    """실제 궤적 좌표(latitude_1 등)는 결과값이므로 쓰지 않고 공항 좌표로 거리를 계산한다."""
    oa = airports().set_index("ident")
    lat1, lon1 = np.radians(sample.origin.map(oa.latitude_deg)), np.radians(sample.origin.map(oa.longitude_deg))
    lat2, lon2 = np.radians(sample.destination.map(oa.latitude_deg)), np.radians(sample.destination.map(oa.longitude_deg))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    sample = sample.assign(distance_km=2 * 6371 * np.arcsin(np.sqrt(a.astype(float))))
    tr, te = sample[sample.period < "202201"], sample[sample.period >= "202201"]
    X_tr, X_te = encode(tr, [te], ["origin", "destination", "typecode"], ["hour", "dow", "distance_km"])
    return model_row("관측 비행 시간(분)", regression_result(X_tr, tr.y.to_numpy(float), X_te, te.y.to_numpy(float)),
                     f"{START_YEAR}-01~2021-12 (월 {SAMPLE_PER_MONTH // 1000}천 편)", f"2022-01~12 (월 {SAMPLE_PER_MONTH // 1000}천 편)", "공항, 기종, 거리, 시각",
                     "ADS-B 첫·마지막 관측 사이 시간이며 예정 시각이 없어 지연 예측은 아닙니다")


def worldbank():
    rows = []
    for code, name in WB.items():
        with zipfile.ZipFile(REF / f"worldbank/{code}.zip") as z:
            raw = z.read(next(n for n in z.namelist() if n.startswith("API_")))
        d = pd.read_csv(io.BytesIO(raw), skiprows=4)
        d = d[d["Country Code"].isin(WB_COUNTRIES)]
        years = [c for c in d.columns if c.isdigit()]
        long = d.melt(id_vars="Country Code", value_vars=years, var_name="연도", value_name="값").dropna()
        rows.append(long.assign(지표=name, 국가=long["Country Code"].map(WB_COUNTRIES), 연도=long.연도.astype(int)).drop(columns="Country Code"))
    wb = pd.concat(rows, ignore_index=True)
    return wb[wb.연도 >= START_YEAR].reset_index(drop=True)


def run(store, paths=None):
    paths = paths or opensky_paths()  # paths: 일부 파일로 빠르게 확인할 때만 지정
    assert paths, "OpenSky 파일 없음"
    with ProcessPoolExecutor(max_workers=2) as ex:  # 파일당 메모리 ~1.5GB
        months = list(ex.map(read_month, paths))
        print("  OpenSky 읽기 완료", flush=True)

    monthly = pd.DataFrame([{"월": f"{m['month'][:4]}-{m['month'][4:]}", "관측 편수": m["total"], "공항 확인 편수": m["known"]} for m in months])
    store.save_table("monthly", monthly)
    country = pd.concat(m["country"] for m in months)
    country["year"] = country.month.str[:4].astype(int)
    by_year = country.groupby(["country", "year"], as_index=False).n.sum()
    top = by_year[by_year.year == 2022].nlargest(15, "n").country
    pivot = by_year[by_year.country.isin(top)].pivot(index="country", columns="year", values="n").sort_values(2022, ascending=False).reset_index()
    pivot.columns = [str(c) for c in pivot.columns]
    store.save_table("country", pivot)
    types = pd.concat(m["type"] for m in months).groupby(["typecode", "year"], as_index=False).n.sum()
    store.save_table("type_top", types[types.year == "2022"].nlargest(20, "n"))
    routes = pd.concat(m["route"] for m in months).groupby(["origin", "destination", "year"], as_index=False).n.sum()
    store.save_table("route_top", routes[routes.year == "2022"].nlargest(20, "n"))
    models = [duration_model(pd.concat(m["sample"] for m in months))]
    del months

    wb = worldbank()
    store.save_table("worldbank", wb)
    ap = airports()
    store.save_table("airport_type", ap.type.map(AIRPORT_TYPES).fillna(ap.type).value_counts().rename("공항 수").rename_axis("종류").reset_index())
    scheduled = ap[(ap.scheduled_service == "yes")]
    store.save_table("airport_country", scheduled.iso_country.value_counts().head(15).rename("공항 수").rename_axis("국가").reset_index())
    aircraft = pd.read_csv(REF / "opensky/aircraftDatabase.csv", usecols=["manufacturername", "built"], dtype="string", low_memory=False)
    store.save_table("manufacturer", aircraft.manufacturername.str.strip().value_counts().head(15).rename("대수").rename_axis("제조사").reset_index())
    routes_of = pd.read_csv(REF / "openflights/routes.dat", header=None, usecols=[0], names=["airline"], dtype="string", na_values="\\N")
    store.save_table("openflights_airline", routes_of.airline.value_counts().head(15).rename("노선 수").rename_axis("항공사").reset_index())

    y2020 = monthly[monthly.월.str.startswith("2020")]["관측 편수"].sum()
    y2022 = monthly[monthly.월.str.startswith("2022")]["관측 편수"].sum()
    world = wb[(wb.국가 == "세계") & (wb.지표 == "여객(명)")].sort_values("연도")
    store.save_summary(
        headline=f"OpenSky 관측 항공편은 2020년 {y2020 / 1e6:,.0f}백만 편에서 2022년 {y2022 / 1e6:,.0f}백만 편으로 {y2022 / y2020 - 1:.0%} 늘었습니다. 세계 항공 여객은 {int(world.연도.iloc[-1])}년 {world.값.iloc[-1] / 1e9:,.1f}십억 명입니다.",
        kpis=[("OpenSky 전체 관측", f"{monthly['관측 편수'].sum() / 1e6:,.0f}백만 편", f"{START_YEAR}-01~2022-12"),
              ("2020년 관측", f"{y2020 / 1e6:,.0f}백만 편", "코로나19로 항공 수요가 크게 줄어든 해입니다"),
              ("2022년 관측", f"{y2022 / 1e6:,.0f}백만 편", f"2020년 대비 {y2022 / y2020 - 1:+.0%}, 수신망 변화도 반영됨"),
              ("공항 수 (OurAirports)", f"{len(ap):,}", "소형·헬기장·폐쇄 포함")],
        models=models,
        notes=["OpenSky는 ADS-B 수신망에 잡힌 편만 포함합니다. 수신망 확대 효과가 섞여 있어 실제 교통량 증감과 다를 수 있습니다.",
               "OpenSky 자료는 비영리·내부 평가 등으로 이용이 제한되고 공유 범위 제한이 있습니다. 공개 전 조건을 확인하세요 (DATA_SOURCES.md).",
               "OpenFlights 노선은 2014년 6월 이후 갱신되지 않았습니다. 연도 구분이 없는 참조 자료(공항, 항공기 등록, OpenFlights)에는 기간 제한을 적용하지 않았습니다.",
               "세계은행 지표는 연 단위입니다. 화물과 운항 지표의 이용 조건은 확인되지 않았습니다."],
        sources=["OpenSky 항공편 목록·항공기 등록", "세계은행(ICAO)", "OurAirports", "OpenFlights"],
    )
