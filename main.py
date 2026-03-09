
import json
import logging
import os
import ast
from pathlib import Path

import pandas as pd
import numpy as np
                                                                            
DATA_DIR = Path(__file__).parent
TLM_FILE = DATA_DIR / "telemetry_data.csv"
TRG_FILE = DATA_DIR / "triggers_soc.csv"
MAP_FILE = DATA_DIR / "vehicle_pnid_mapping.csv"
SYN_FILE = DATA_DIR / "artificial_ign_off_data.json"

OUTPUT_DIR = DATA_DIR / "output"
IGNITION_EVENTS_OUT = OUTPUT_DIR / "IgnitionEvents.csv"
CHARGING_EVENTS_OUT = OUTPUT_DIR / "ChargingEvents.csv"
CHARGING_STATUS_EVENTS_OUT = OUTPUT_DIR / "ChargingStatusEvents.csv"
DATA_QUALITY_REPORT = OUTPUT_DIR / "data_quality_report.txt"

BATTERY_WINDOW_SEC = 300                                                                                       
CHARGE_THRESHOLD_IGN_OFF = 2.0                                                  
CHARGE_THRESHOLD_IGN_ON  = 5.0                                                        
MIN_CHARGE_DURATION_SEC  = 60                                             
MERGE_GAP_SEC            = 300                                                 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("charge_analytics")
                                                                                                                                                                              
class DataQualityReport:
                                                                 

    def __init__(self):
        self.findings: list[str] = []

    def add(self, msg: str):
        self.findings.append(msg)
        log.warning("DQ  %s", msg)

    def write(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write("DATA QUALITY REPORT\n")
            f.write("=" * 60 + "\n\n")
            for i, finding in enumerate(self.findings, 1):
                f.write(f"{i}. {finding}\n\n")
        log.info("Data-quality report saved to %s", path)


dq = DataQualityReport()


def load_tlm() -> pd.DataFrame:
                                            
    log.info("Loading TLM (%s) …", TLM_FILE.name)
    df = pd.read_csv(TLM_FILE)
    df.drop(columns=["Unnamed: 0"], errors="ignore", inplace=True)

                      
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], errors="coerce")
    bad_ts = df["TIMESTAMP"].isna().sum()
    if bad_ts:
        dq.add(f"TLM: {bad_ts} rows with unparseable timestamps — dropped.")
        df.dropna(subset=["TIMESTAMP"], inplace=True)

                                            
    df["IGNITION_STATUS"] = df["IGNITION_STATUS"].str.strip().str.lower()
    unknown_ign = (df["IGNITION_STATUS"] == "unknown").sum()
    if unknown_ign:
        dq.add(
            f"TLM: {unknown_ign} rows have IGNITION_STATUS='unknown' — "
            "treated as missing."
        )
        df.loc[df["IGNITION_STATUS"] == "unknown", "IGNITION_STATUS"] = np.nan

                                     
    outlier_battery = (df["EV_BATTERY_LEVEL"] > 100).sum()
    if outlier_battery:
        dq.add(
            f"TLM: {outlier_battery} battery readings > 100 % — "
            "capped at 100."
        )
        df.loc[df["EV_BATTERY_LEVEL"] > 100, "EV_BATTERY_LEVEL"] = 100.0

    neg_battery = (df["EV_BATTERY_LEVEL"] < 0).sum()
    if neg_battery:
        dq.add(f"TLM: {neg_battery} negative battery readings — set to NaN.")
        df.loc[df["EV_BATTERY_LEVEL"] < 0, "EV_BATTERY_LEVEL"] = np.nan

    neg_speed = (df["SPEED"] < 0).sum()
    if neg_speed:
        dq.add(f"TLM: {neg_speed} negative speed readings — set to 0.")
        df.loc[df["SPEED"] < 0, "SPEED"] = 0.0

                                        
    dup_ids = df["ID"].duplicated().sum()
    if dup_ids:
        dq.add(f"TLM: {dup_ids} duplicate row IDs — deduplicated.")
        df.drop_duplicates(subset=["ID"], keep="first", inplace=True)

                     
    total = len(df)
    for col in ["SPEED", "IGNITION_STATUS", "EV_BATTERY_LEVEL", "ODOMETER"]:
        missing = df[col].isna().sum()
        pct = 100 * missing / total
        if pct > 50:
            dq.add(
                f"TLM: Column '{col}' is {pct:.1f}% null "
                f"({missing:,}/{total:,}) — sparse but usable."
            )

    df.sort_values(["VEHICLE_ID", "TIMESTAMP"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    log.info("TLM loaded: %s rows, %d vehicles.", f"{len(df):,}", df["VEHICLE_ID"].nunique())
    return df


def load_trg() -> pd.DataFrame:
                                             
    log.info("Loading TRG (%s) …", TRG_FILE.name)
    df = pd.read_csv(TRG_FILE)
    df.drop(columns=["Unnamed: 0"], errors="ignore", inplace=True)

                                                
    df["CTS"] = df["CTS"].str.replace(r"\s*IST\+0530", "+05:30", regex=True)
    df["CTS"] = pd.to_datetime(df["CTS"], errors="coerce", utc=True)
                                                                           
                                                                                
    df["CTS"] = df["CTS"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)

    bad_ts = df["CTS"].isna().sum()
    if bad_ts:
        dq.add(f"TRG: {bad_ts} rows with unparseable timestamps — dropped.")
        df.dropna(subset=["CTS"], inplace=True)

                          
    valid_names = {"IGN_CYL", "CHARGE_STATE", "EV_CHARGE_STATE"}
    bad_names = ~df["NAME"].isin(valid_names)
    if bad_names.sum():
        dq.add(
            f"TRG: {bad_names.sum()} rows with unexpected NAME values: "
            f"{df.loc[bad_names, 'NAME'].unique().tolist()} — dropped."
        )
        df = df[~bad_names]

                             
    ign_mask = df["NAME"] == "IGN_CYL"
    bad_ign = ign_mask & ~df["VAL"].isin(["ON", "OFF"])
    if bad_ign.sum():
        dq.add(
            f"TRG: {bad_ign.sum()} IGN_CYL rows with unexpected VAL "
            f"{df.loc[bad_ign, 'VAL'].unique().tolist()} — dropped."
        )
        df = df[~bad_ign]

                                      
    cs_mask = df["NAME"] == "CHARGE_STATE"
    df.loc[cs_mask, "VAL_NUM"] = pd.to_numeric(df.loc[cs_mask, "VAL"], errors="coerce")
    bad_cs = cs_mask & df["VAL_NUM"].isna()
    if bad_cs.sum():
        dq.add(
            f"TRG: {bad_cs.sum()} CHARGE_STATE rows with non-numeric VAL — dropped."
        )
        df = df[~bad_cs]

                                     
    evcs_mask = df["NAME"] == "EV_CHARGE_STATE"
    valid_charge_states = {"Active", "Aborted", "Complete", "Completed"}
    bad_evcs = evcs_mask & ~df["VAL"].isin(valid_charge_states)
    if bad_evcs.sum():
        dq.add(
            f"TRG: {bad_evcs.sum()} EV_CHARGE_STATE rows with unexpected VAL "
            f"{df.loc[bad_evcs, 'VAL'].unique().tolist()} — dropped."
        )
        df = df[~bad_evcs]

                              
    dup_rows = df.duplicated(subset=["CTS", "PNID", "NAME", "VAL"]).sum()
    if dup_rows:
        dq.add(f"TRG: {dup_rows} exact duplicate rows — deduplicated.")
        df.drop_duplicates(subset=["CTS", "PNID", "NAME", "VAL"], keep="first", inplace=True)

    df.sort_values(["PNID", "CTS"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    log.info("TRG loaded: %s rows, %d PNIDs.", f"{len(df):,}", df["PNID"].nunique())
    return df


def load_map() -> dict[str, list[int]]:
                                                                       
    log.info("Loading MAP (%s) …", MAP_FILE.name)
    df = pd.read_csv(MAP_FILE)
    df.drop(columns=["Unnamed: 0"], errors="ignore", inplace=True)

                                                                            
    df.drop_duplicates(subset=["ID"], keep="last", inplace=True)

    mapping: dict[str, list[int]] = {}
    for _, row in df.iterrows():
        vid = row["ID"]
        ids_str = row["IDS"]
        if pd.isna(ids_str):
            dq.add(f"MAP: Vehicle {vid} has no PNIDs — will get events only from TLM/SYN.")
            mapping[vid] = []
            continue
        try:
            pnids = json.loads(ids_str)
            mapping[vid] = [int(p) for p in pnids]
        except (json.JSONDecodeError, ValueError):
            try:
                pnids = ast.literal_eval(ids_str)
                mapping[vid] = [int(p) for p in pnids]
            except Exception:
                dq.add(f"MAP: Could not parse PNIDs for vehicle {vid}: {ids_str!r}")
                mapping[vid] = []

                                          
    pnid_to_vehicle: dict[int, str] = {}
    for vid, pnids in mapping.items():
        for pnid in pnids:
            if pnid in pnid_to_vehicle:
                dq.add(
                    f"MAP: PNID {pnid} maps to multiple vehicles: "
                    f"{pnid_to_vehicle[pnid]} and {vid}. Using latest."
                )
            pnid_to_vehicle[pnid] = vid

    log.info("MAP loaded: %d vehicles, %d total PNIDs.", len(mapping), len(pnid_to_vehicle))
    return mapping, pnid_to_vehicle


def load_syn() -> pd.DataFrame:
                                                
    log.info("Loading SYN (%s) …", SYN_FILE.name)
    with open(SYN_FILE) as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df.rename(columns={"vehicleId": "vehicle_id", "timestamp": "event_ts"}, inplace=True)
    df["event_ts"] = pd.to_datetime(df["event_ts"], errors="coerce", utc=True)
                                           
    df["event_ts"] = df["event_ts"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)

    bad_ts = df["event_ts"].isna().sum()
    if bad_ts:
        dq.add(f"SYN: {bad_ts} rows with unparseable timestamps — dropped.")
        df.dropna(subset=["event_ts"], inplace=True)

    log.info("SYN loaded: %d events, %d vehicles.", len(df), df["vehicle_id"].nunique())
    return df

def extract_ignition_from_tlm(tlm: pd.DataFrame) -> pd.DataFrame:
       
    log.info("Extracting ignition events from TLM …")
                                                 
    df = tlm.dropna(subset=["IGNITION_STATUS"]).copy()
    df = df[df["IGNITION_STATUS"].isin(["on", "off"])].copy()
    df.sort_values(["VEHICLE_ID", "TIMESTAMP"], inplace=True)

                                                                               
    df["prev_ign"] = df.groupby("VEHICLE_ID")["IGNITION_STATUS"].shift(1)
                                                                              
    events = df[df["IGNITION_STATUS"] != df["prev_ign"]].copy()

    events["event"] = events["IGNITION_STATUS"].map(
        {"on": "ignitionOn", "off": "ignitionOff"}
    )
    events = events.rename(columns={
        "VEHICLE_ID": "vehicle_id",
        "TIMESTAMP": "event_ts",
    })[["vehicle_id", "event", "event_ts"]].copy()

    events["source"] = "TLM"
    log.info("  TLM ignition events: %d", len(events))
    return events


def extract_ignition_from_trg(
    trg: pd.DataFrame, pnid_to_vehicle: dict[int, str]
) -> pd.DataFrame:

       
    log.info("Extracting ignition events from TRG …")
    df = trg[trg["NAME"] == "IGN_CYL"].copy()
    df["vehicle_id"] = df["PNID"].map(pnid_to_vehicle)

    unmapped = df["vehicle_id"].isna().sum()
    if unmapped:
        unmapped_pnids = df.loc[df["vehicle_id"].isna(), "PNID"].unique()
        dq.add(
            f"TRG→IGN: {unmapped} IGN_CYL rows with unmapped PNIDs: "
            f"{unmapped_pnids.tolist()} — dropped."
        )
        df.dropna(subset=["vehicle_id"], inplace=True)

    df.sort_values(["vehicle_id", "CTS"], inplace=True)

                        
    df["prev_val"] = df.groupby("vehicle_id")["VAL"].shift(1)
    events = df[df["VAL"] != df["prev_val"]].copy()

    events["event"] = events["VAL"].map({"ON": "ignitionOn", "OFF": "ignitionOff"})
    events = events.rename(columns={"CTS": "event_ts"})[
        ["vehicle_id", "event", "event_ts"]
    ].copy()
    events["source"] = "TRG"
    log.info("  TRG ignition events: %d", len(events))
    return events


def extract_ignition_from_syn(syn: pd.DataFrame) -> pd.DataFrame:
                                                   
    log.info("Extracting ignition events from SYN …")
    events = syn[["vehicle_id", "event_ts"]].copy()
    events["event"] = "ignitionOff"
    events["source"] = "SYN"
    log.info("  SYN ignition events: %d", len(events))
    return events


def fuse_ignition_events(
    tlm_events: pd.DataFrame,
    trg_events: pd.DataFrame,
    syn_events: pd.DataFrame,
) -> pd.DataFrame:
       
    log.info("Fusing ignition events from all sources …")
    all_events = pd.concat([tlm_events, trg_events, syn_events], ignore_index=True)
    all_events.sort_values(["vehicle_id", "event_ts"], inplace=True)
    all_events.reset_index(drop=True, inplace=True)                                                 
    source_priority = {"TLM": 0, "TRG": 1, "SYN": 2}
    all_events["_priority"] = all_events["source"].map(source_priority)
    all_events.sort_values(
        ["vehicle_id", "event_ts", "_priority"], inplace=True
    )

    deduped = []
    prev_vid = None
    prev_event = None
    prev_ts = None

    for _, row in all_events.iterrows():
        vid = row["vehicle_id"]
        evt = row["event"]
        ts = row["event_ts"]

        if vid == prev_vid and evt == prev_event and prev_ts is not None:
            diff = abs((ts - prev_ts).total_seconds())
            if diff < 10:
                                                                         
                continue

        deduped.append(row)
        prev_vid = vid
        prev_event = evt
        prev_ts = ts

    result = pd.DataFrame(deduped)
    result = result[["vehicle_id", "event", "event_ts", "source"]].copy()
    result.sort_values(["vehicle_id", "event_ts"], inplace=True)
    result.reset_index(drop=True, inplace=True)

    dup_count = len(all_events) - len(result)
    if dup_count:
        dq.add(
            f"Ignition fusion: {dup_count} near-duplicate events removed "
            "(same vehicle + event within 10 s)."
        )

    log.info("Fused ignition events: %d total.", len(result))
    return result


def extract_charging_status_events(
    trg: pd.DataFrame, pnid_to_vehicle: dict[int, str]
) -> pd.DataFrame:

       
    log.info("Extracting charging status events from TRG …")
    df = trg[trg["NAME"] == "EV_CHARGE_STATE"].copy()
    df["vehicle_id"] = df["PNID"].map(pnid_to_vehicle)

    unmapped = df["vehicle_id"].isna().sum()
    if unmapped:
        unmapped_pnids = df.loc[df["vehicle_id"].isna(), "PNID"].unique()
        dq.add(
            f"TRG→ChargeStatus: {unmapped} EV_CHARGE_STATE rows with unmapped PNIDs: "
            f"{unmapped_pnids.tolist()} — dropped."
        )
        df.dropna(subset=["vehicle_id"], inplace=True)

                                                      
    df["event"] = df["VAL"].replace({"Aborted": "Abort", "Completed": "Complete"})

    result = df.rename(columns={"CTS": "event_ts"})[
        ["vehicle_id", "event", "event_ts"]
    ].copy()
    result.sort_values(["vehicle_id", "event_ts"], inplace=True)
    result.reset_index(drop=True, inplace=True)
    log.info("Charging status events: %d", len(result))
    return result
                                                                            
def build_battery_timeline(
    tlm: pd.DataFrame,
    trg: pd.DataFrame,
    pnid_to_vehicle: dict[int, str],
) -> pd.DataFrame:
       
    log.info("Building battery timeline …")
              
    tlm_batt = tlm.dropna(subset=["EV_BATTERY_LEVEL"])[
        ["VEHICLE_ID", "TIMESTAMP", "EV_BATTERY_LEVEL"]
    ].copy()
    tlm_batt.rename(
        columns={
            "VEHICLE_ID": "vehicle_id",
            "TIMESTAMP": "ts",
            "EV_BATTERY_LEVEL": "battery_pct",
        },
        inplace=True,
    )
    tlm_batt["source"] = "TLM"

                             
    trg_cs = trg[trg["NAME"] == "CHARGE_STATE"].copy()
    trg_cs["vehicle_id"] = trg_cs["PNID"].map(pnid_to_vehicle)
    trg_cs.dropna(subset=["vehicle_id"], inplace=True)
    trg_cs["battery_pct"] = pd.to_numeric(trg_cs["VAL"], errors="coerce")
    trg_cs.dropna(subset=["battery_pct"], inplace=True)
    trg_cs = trg_cs.rename(columns={"CTS": "ts"})[
        ["vehicle_id", "ts", "battery_pct"]
    ].copy()
    trg_cs["source"] = "TRG"

    timeline = pd.concat([tlm_batt, trg_cs], ignore_index=True)
                
    timeline.loc[timeline["battery_pct"] > 100, "battery_pct"] = 100.0
    timeline.sort_values(["vehicle_id", "ts"], inplace=True)
    timeline.reset_index(drop=True, inplace=True)

    log.info(
        "Battery timeline: %s readings across %d vehicles.",
        f"{len(timeline):,}",
        timeline["vehicle_id"].nunique(),
    )
    return timeline

def associate_battery(
    events: pd.DataFrame,
    battery_timeline: pd.DataFrame,
    window_sec: int = BATTERY_WINDOW_SEC,
) -> pd.DataFrame:
   
    log.info("Associating battery levels (±%d s window) …", window_sec)
    events = events.copy()
    events["battery_pct"] = np.nan
    events["battery_ts"] = pd.NaT

    for vid in events["vehicle_id"].unique():
        batt = battery_timeline[battery_timeline["vehicle_id"] == vid].copy()
        if batt.empty:
            continue
        batt_ts = batt["ts"].values                          
        batt_pct = batt["battery_pct"].values

        evt_mask = events["vehicle_id"] == vid
        evt_idx = events.index[evt_mask]
        evt_ts = events.loc[evt_idx, "event_ts"].values

        for i, ets in zip(evt_idx, evt_ts):
            diffs = (batt_ts - ets).astype("timedelta64[s]").astype(float)
            within = np.abs(diffs) <= window_sec
            if not within.any():
                continue

            abs_diffs = np.abs(diffs[within])
            signs = np.sign(diffs[within])                    
            candidates_pct = batt_pct[within]
            candidates_ts_arr = batt_ts[within]                                                                 
            order = np.lexsort((-signs, abs_diffs))
            best = order[0]

            events.at[i, "battery_pct"] = candidates_pct[best]
            events.at[i, "battery_ts"] = pd.Timestamp(candidates_ts_arr[best])

    associated = events["battery_pct"].notna().sum()
    total = len(events)
    log.info(
        "Battery associated: %d / %d events (%.1f%%).",
        associated, total, 100 * associated / total if total else 0,
    )
    return events
                                                                                                                                                                                       
def detect_charging_events(
    ignition_events: pd.DataFrame,
    charging_status_events: pd.DataFrame,
    battery_timeline: pd.DataFrame,
) -> pd.DataFrame:
       
    log.info("Detecting charging events …")

                                                     
    ign = ignition_events[["vehicle_id", "event", "event_ts", "battery_pct"]].copy()
    ign["event_type"] = "ignition"

    cs = charging_status_events.copy()
    if "battery_pct" not in cs.columns:
        cs["battery_pct"] = np.nan
    cs["event_type"] = "charge_status"

    combined = pd.concat(
        [
            ign.rename(columns={"event_ts": "ts"}),
            cs.rename(columns={"event_ts": "ts"}),
        ],
        ignore_index=True,
    )
    combined.sort_values(["vehicle_id", "ts"], inplace=True)

    charging_events = []

    for vid in combined["vehicle_id"].unique():
        vdf = combined[combined["vehicle_id"] == vid].reset_index(drop=True)
        batt = battery_timeline[battery_timeline["vehicle_id"] == vid].copy()

        if len(vdf) < 2 or batt.empty:
            continue

                                                
                                                          
        current_ign = None
        sessions = []

        for idx in range(len(vdf) - 1):
            row = vdf.iloc[idx]
            next_row = vdf.iloc[idx + 1]

                                   
            if row["event_type"] == "ignition":
                current_ign = row["event"]

            batt_start = row["battery_pct"]
            batt_end = next_row["battery_pct"]

            if pd.isna(batt_start) or pd.isna(batt_end):
                continue

            delta_pct = batt_end - batt_start
            delta_sec = (next_row["ts"] - row["ts"]).total_seconds()

            if delta_sec <= 0:
                continue

                                                      
            if current_ign == "ignitionOn":
                threshold = CHARGE_THRESHOLD_IGN_ON
            else:
                threshold = CHARGE_THRESHOLD_IGN_OFF

            if delta_pct >= threshold:
                sessions.append({
                    "vehicle_id": vid,
                    "start_ts": row["ts"],
                    "end_ts": next_row["ts"],
                    "delta_battery_pct": round(delta_pct, 2),
                    "start_battery_pct": round(batt_start, 2),
                    "end_battery_pct": round(batt_end, 2),
                    "duration_sec": delta_sec,
                    "ignition_state": current_ign or "unknown",
                })

                                                                   
        merged = _merge_charge_sessions(sessions)
        charging_events.extend(merged)

    result = pd.DataFrame(charging_events)
    if result.empty:
        result = pd.DataFrame(columns=[
            "vehicle_id", "start_ts", "end_ts", "delta_battery_pct",
            "start_battery_pct", "end_battery_pct", "duration_sec",
            "ignition_state",
        ])

    log.info("Charging events detected: %d", len(result))
    return result


def _merge_charge_sessions(sessions: list[dict]) -> list[dict]:
                                                                             
    if not sessions:
        return []

    merged = [sessions[0]]
    for s in sessions[1:]:
        prev = merged[-1]
        gap = (s["start_ts"] - prev["end_ts"]).total_seconds()

        if gap <= MERGE_GAP_SEC and s["vehicle_id"] == prev["vehicle_id"]:
                                         
            prev["end_ts"] = s["end_ts"]
            prev["end_battery_pct"] = s["end_battery_pct"]
            prev["delta_battery_pct"] = round(
                prev["end_battery_pct"] - prev["start_battery_pct"], 2
            )
            prev["duration_sec"] = (
                prev["end_ts"] - prev["start_ts"]
            ).total_seconds()
        else:
            merged.append(s)

                                                       
    merged = [s for s in merged if s["duration_sec"] >= MIN_CHARGE_DURATION_SEC]
    return merged
                                                                   
def main():
    log.info("=" * 60)
    log.info("Vehicle Event & Charge-Analytics Pipeline")
    log.info("=" * 60)

                                                                       
    tlm = load_tlm()
    trg = load_trg()
    vehicle_to_pnids, pnid_to_vehicle = load_map()
    syn = load_syn()

                                                                       
    ign_tlm = extract_ignition_from_tlm(tlm)
    ign_trg = extract_ignition_from_trg(trg, pnid_to_vehicle)
    ign_syn = extract_ignition_from_syn(syn)
    ignition_events = fuse_ignition_events(ign_tlm, ign_trg, ign_syn)                                                                   
    charging_status_events = extract_charging_status_events(trg, pnid_to_vehicle)                                                                   
    battery_timeline = build_battery_timeline(tlm, trg, pnid_to_vehicle)
    ignition_events = associate_battery(ignition_events, battery_timeline)
    charging_status_events = associate_battery(
        charging_status_events, battery_timeline
    )

                                                                       
    charging_events = detect_charging_events(
        ignition_events, charging_status_events, battery_timeline
    )

                                                                      
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

                                                                             
    ign_out = ignition_events[["vehicle_id", "event", "event_ts"]].copy()
    ign_out.to_csv(IGNITION_EVENTS_OUT, index=False)
    log.info("Saved %s (%d rows)", IGNITION_EVENTS_OUT.name, len(ign_out))

                           
    charging_events.to_csv(CHARGING_EVENTS_OUT, index=False)
    log.info("Saved %s (%d rows)", CHARGING_EVENTS_OUT.name, len(charging_events))

                                                                   
    cs_out = charging_status_events[["vehicle_id", "event", "event_ts"]].copy()
    cs_out.to_csv(CHARGING_STATUS_EVENTS_OUT, index=False)
    log.info("Saved %s (%d rows)", CHARGING_STATUS_EVENTS_OUT.name, len(cs_out))

                         
    dq.write(DATA_QUALITY_REPORT)

    log.info("=" * 60)
    log.info("Pipeline complete. Outputs in %s", OUTPUT_DIR)
    log.info("=" * 60)                     
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Ignition Events:         {len(ign_out):>8,}")
    print(f"  Charging Status Events:  {len(cs_out):>8,}")
    print(f"  Charging Events:         {len(charging_events):>8,}")
    print(f"  Data Quality Findings:   {len(dq.findings):>8}")
    print(f"\n  Outputs written to: {OUTPUT_DIR}/")
    print("=" * 60)

if __name__ == "__main__":
    main()
