#!/usr/bin/env python3
"""
OpenSky Flight Data Fetcher (Historical /states/all)
Fetches state vectors from OpenSky Network, saves them to a single-file per-bbox cache,
and tracks detailed metrics.

Usage:
    python fetch_flight_data.py [--lookback-min 20] [--step-sec 15]
"""

import os
import sys
import json
import time
import hashlib
import argparse
import requests
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any, Tuple, Set

try:
    import gc
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tqdm import tqdm
except ImportError as e:
    sys.exit(f"❌ Missing dependency: {e}. Please ensure you are using the project virtualenv.")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE_DIR = os.path.join(SCRIPT_DIR, "flight_data_cache")
DEFAULT_CREDS_FILE = os.path.join(SCRIPT_DIR, "credentials.json")


class OpenSkyTokenManager:
    """Handles OpenSky OAuth2 token exchange and automatic proactive refresh."""
    TOKEN_URL = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
    TOKEN_REFRESH_MARGIN = 30  # Refresh 30 seconds prior to expiry

    def __init__(self, client_id: str = "", client_secret: str = "", credentials_file: str = DEFAULT_CREDS_FILE):
        self.client_id = client_id.strip() or os.environ.get("OPENSKY_CLIENT_ID", "").strip()
        self.client_secret = client_secret.strip() or os.environ.get("OPENSKY_CLIENT_SECRET", "").strip()
        if not (self.client_id and self.client_secret) and os.path.exists(credentials_file):
            try:
                with open(credentials_file, 'r', encoding='utf-8') as f:
                    _c = json.load(f)
                    self.client_id     = (self.client_id or _c.get("clientId") or _c.get("client_id") or "").strip()
                    self.client_secret = (self.client_secret or _c.get("clientSecret") or _c.get("client_secret") or "").strip()
            except Exception as e:
                print(f"⚠️ Could not load credentials from {credentials_file}: {e}")
                
        self.token = None
        self.expires_at = None

    @property
    def is_configured(self) -> bool:
        return bool(self.client_id and self.client_secret)

    def get_token(self) -> Optional[str]:
        """Returns a valid OAuth2 Bearer token, refreshing if necessary."""
        if not self.is_configured:
            return None
        if self.token and self.expires_at and datetime.now() < self.expires_at:
            return self.token
        return self._refresh()

    def _refresh(self) -> Optional[str]:
        """Requests a new access token using client_credentials flow."""
        try:
            r = requests.post(
                self.TOKEN_URL,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                },
                timeout=10
            )
            if r.status_code == 200:
                data = r.json()
                self.token = data["access_token"]
                expires_in = data.get("expires_in", 1800)
                self.expires_at = datetime.now() + timedelta(seconds=expires_in - self.TOKEN_REFRESH_MARGIN)
                return self.token
            else:
                print(f"⚠️ OpenSky OAuth2 Authentication Failed (HTTP {r.status_code}): {r.text[:120]}")
                return None
        except Exception as e:
            print(f"❌ Failed to reach OpenSky Auth server: {e}")
            return None

    def headers(self) -> Dict[str, str]:
        """Returns authorization headers if configured."""
        token = self.get_token()
        headers = {
            "User-Agent": "OpenSkyFlightDataDaemon/2.0 (Automated Ingestion)"
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers


class OpenSkyBBoxCache:
    """Manages a single persistent Parquet cache file per bounding box with clean coordinate filename."""
    def __init__(self, cache_dir: str = DEFAULT_CACHE_DIR):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def get_cache_path(self, lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> str:
        lat_min_s = f"{lat_min:.4f}".replace(".", "_")
        lat_max_s = f"{lat_max:.4f}".replace(".", "_")
        lon_min_s = f"{lon_min:.4f}".replace(".", "_")
        lon_max_s = f"{lon_max:.4f}".replace(".", "_")
        fname = f"bbox-lat-{lat_min_s}-{lat_max_s}-lon-{lon_min_s}-{lon_max_s}.parquet"
        return os.path.join(self.cache_dir, fname)
        
    def load(self, lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> Tuple[pd.DataFrame, Set[int]]:
        """Loads all cached trajectory points and the set of already-queried timestamps for this bbox."""
        path = self.get_cache_path(lat_min, lat_max, lon_min, lon_max)
        if not os.path.exists(path):
            return pd.DataFrame(), set()
        try:
            table = pq.read_table(path)
            df = table.to_pandas()
            del table
            if not df.empty:
                for c in ['latitude', 'longitude', 'altitude', 'velocity', 'vertical_rate', 'track']:
                    if c in df.columns:
                        df[c] = df[c].astype(np.float32)
            meta = pq.read_metadata(path).metadata or {}
            raw_queried = meta.get(b'queried_ts', b'[]').decode('utf-8')
            queried_ts = set(json.loads(raw_queried))
            if not queried_ts and not df.empty and 'timestamp' in df.columns:
                queried_ts = set(df['timestamp'].astype(int).unique())
            gc.collect()
            return df, queried_ts
        except Exception as e:
            print(f"⚠️ Could not read cache {os.path.basename(path)}: {e}")
            return pd.DataFrame(), set()
            
    def save(self, lat_min: float, lat_max: float, lon_min: float, lon_max: float, df_combined: pd.DataFrame, queried_ts: Set[int]):
        """Saves the combined dataframe and updated queried timestamps to the single bbox parquet file."""
        path = self.get_cache_path(lat_min, lat_max, lon_min, lon_max)
        try:
            if df_combined.empty:
                df_combined = pd.DataFrame(columns=['icao24', 'callsign', 'timestamp', 'latitude', 'longitude', 'altitude', 'velocity', 'track', 'vertical_rate', 'on_ground'])
            else:
                df_combined = df_combined.drop_duplicates(subset=['icao24', 'timestamp', 'latitude', 'longitude'])
                
            table = pa.Table.from_pandas(df_combined)
            meta = table.schema.metadata or {}
            meta[b'queried_ts'] = json.dumps(sorted(list(queried_ts))).encode('utf-8')
            table = table.replace_schema_metadata(meta)
            pq.write_table(table, path)
        except Exception as e:
            print(f"⚠️ Could not write cache file {os.path.basename(path)}: {e}")


class OpenSkyStatesFetcher:
    """Queries OpenSky Network historical state vectors exclusively via /states/all."""
    BASE_URL = "https://opensky-network.org/api"
    
    def __init__(self, client_id: str = "", client_secret: str = "", cache_dir: str = DEFAULT_CACHE_DIR, credentials_file: str = DEFAULT_CREDS_FILE):
        self.token_manager = OpenSkyTokenManager(client_id, client_secret, credentials_file)
        self.cache = OpenSkyBBoxCache(cache_dir)
        self.session = requests.Session()
        
    def fetch_states(
        self,
        lat_min: float, lat_max: float, lon_min: float, lon_max: float,
        start_ts: int, end_ts: int,
        time_step_sec: int = 15,
        use_cache: bool = True,
        active_start_hour: int = 5,
        active_end_hour: int = 23,
        use_local_tz: bool = True
    ) -> pd.DataFrame:
        """Queries OpenSky /states/all across [start_ts, end_ts] within operating hours (5 AM - 11 PM), updating cache only for successful 200 OK responses."""
        start_ts = start_ts - (start_ts % 60)
        end_ts   = end_ts - (end_ts % 60)
        
        dt_start_str = datetime.fromtimestamp(start_ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        dt_end_str   = datetime.fromtimestamp(end_ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        cache_path = self.cache.get_cache_path(lat_min, lat_max, lon_min, lon_max)
        
        tz_name = datetime.now().astimezone().tzname() if use_local_tz else "UTC"
        print(f"🌐 Querying OpenSky /states/all for historical time range: {dt_start_str} -> {dt_end_str}")
        print(f"📍 Bounding Box: [{lat_min:.4f}, {lat_max:.4f}] x [{lon_min:.4f}, {lon_max:.4f}]")
        print(f"💾 Bbox Cache File: {os.path.basename(cache_path)}")
        print(f"🕒 Operating Hours: {active_start_hour:02d}:00 -> {active_end_hour:02d}:00 ({tz_name})")
        
        if self.token_manager.is_configured:
            print(f"🔑 Using OpenSky OAuth2 Client Credentials (Client: {self.token_manager.client_id}).")
        else:
            print("👤 Running in Anonymous Mode (400 daily credits quota).")
            
        df_bbox, queried_ts = self.cache.load(lat_min, lat_max, lon_min, lon_max) if use_cache else (pd.DataFrame(), set())
        
        all_timestamps = list(range(start_ts, end_ts + 1, time_step_sec))
        if not all_timestamps:
            all_timestamps = [start_ts]
            
        # Filter timestamps to operating hours (5:00 AM to 11:00 PM)
        def is_within_operating_hours(t_epoch: int) -> bool:
            dt = datetime.fromtimestamp(t_epoch, tz=timezone.utc)
            if use_local_tz:
                dt = dt.astimezone()
            return active_start_hour <= dt.hour < active_end_hour

        timestamps = [t for t in all_timestamps if is_within_operating_hours(t)]
        
        if not timestamps:
            print(f"\n🌙 Outside operating hours ({active_start_hour:02d}:00 - {active_end_hour:02d}:00 {tz_name}).")
            print("💤 Skipping API calls during nighttime curfew.")
            if not df_bbox.empty:
                df_res = df_bbox[
                    (df_bbox['timestamp'] >= start_ts - 5) & 
                    (df_bbox['timestamp'] <= end_ts + 5)
                ].copy()
            else:
                df_res = pd.DataFrame()
            return df_res
            
        print(f"⏱️  Processing {len(timestamps)} active time steps out of {len(all_timestamps)} in window (every {time_step_sec}s)...")
        
        margin = 0.35
        new_records = []
        states_url = f"{self.BASE_URL}/states/all"
        
        cache_hits = 0
        cache_misses = 0
        successful_api = 0
        failed_api = 0
        failure_reasons = {}
        credits_remaining = None
        
        for ts in tqdm(timestamps, desc="Fetching /states/all"):
            ts_aligned = ts - (ts % 5)
            
            if use_cache and ts_aligned in queried_ts:
                cache_hits += 1
                continue
                
            cache_misses += 1
            params = {
                "time": ts_aligned,
                "lamin": round(lat_min - margin, 4),
                "lamax": round(lat_max + margin, 4),
                "lomin": round(lon_min - margin, 4),
                "lomax": round(lon_max + margin, 4)
            }
            
            try:
                headers = self.token_manager.headers()
                r = self.session.get(states_url, params=params, headers=headers, timeout=12)
                if "X-Rate-Limit-Remaining" in r.headers:
                    credits_remaining = r.headers.get("X-Rate-Limit-Remaining")
                    
                if r.status_code == 200:
                    successful_api += 1
                    states = r.json().get('states', []) or []
                    for s in states:
                        if s[6] is not None and s[5] is not None:
                            alt = float(s[7]) if s[7] is not None else (float(s[13]) if s[13] is not None else 0.0)
                            cat = int(s[17]) if len(s) > 17 and s[17] is not None else 0
                            new_records.append({
                                "icao24": s[0],
                                "callsign": (s[1] or 'N/A').strip(),
                                "timestamp": ts_aligned,
                                "latitude": float(s[6]),
                                "longitude": float(s[5]),
                                "altitude": alt,
                                "velocity": float(s[9]) if s[9] is not None else 0.0,
                                "track": float(s[10]) if s[10] is not None else 0.0,
                                "vertical_rate": float(s[11]) if s[11] is not None else 0.0,
                                "on_ground": bool(s[8]),
                                "category": cat
                            })
                    queried_ts.add(ts_aligned)
                elif r.status_code in [400, 403]:
                    failed_api += 1
                    reason = f"HTTP {r.status_code} (State vectors unavailable for time > 1h old)"
                    failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
                elif r.status_code == 429:
                    failed_api += 1
                    reason = "HTTP 429 (Rate Limit / Quota Exceeded)"
                    failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
                    time.sleep(2)
                elif r.status_code >= 500:
                    failed_api += 1
                    reason = f"HTTP {r.status_code} (OpenSky Server Error)"
                    failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
                else:
                    failed_api += 1
                    reason = f"HTTP {r.status_code} ({r.text[:60]})"
                    failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            except Exception as e:
                failed_api += 1
                reason = f"Network/Connection Error ({type(e).__name__})"
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
                
            time.sleep(0.04)
            
        if (new_records or successful_api > 0) and use_cache:
            if new_records:
                df_new = pd.DataFrame(new_records)
                if not df_bbox.empty:
                    df_combined = pd.concat([df_bbox, df_new], ignore_index=True)
                else:
                    df_combined = df_new
            else:
                df_combined = df_bbox
            self.cache.save(lat_min, lat_max, lon_min, lon_max, df_combined, queried_ts)
            df_bbox = df_combined
            
        if not df_bbox.empty:
            df_res = df_bbox[
                (df_bbox['timestamp'] >= start_ts - 5) & 
                (df_bbox['timestamp'] <= end_ts + 5)
            ].copy()
            df_res = df_res.drop_duplicates(subset=['icao24', 'timestamp', 'latitude', 'longitude'])
            df_res = df_res[
                (df_res['latitude'] >= lat_min - margin) &
                (df_res['latitude'] <= lat_max + margin) &
                (df_res['longitude'] >= lon_min - margin) &
                (df_res['longitude'] <= lon_max + margin)
            ]
        else:
            df_res = pd.DataFrame()
            
        total_steps = len(timestamps)
        hit_pct = (cache_hits / total_steps * 100) if total_steps > 0 else 0
        miss_pct = (cache_misses / total_steps * 100) if total_steps > 0 else 0
        
        print()
        print("=" * 60)
        print("📈 INGESTION & CACHE PERFORMANCE METRICS")
        print("=" * 60)
        print(f"  • Bbox Cache File            : {os.path.basename(cache_path)}")
        print(f"  • Total Time Steps Processed : {total_steps}")
        print(f"  • Cache Hits (Found in File) : {cache_hits} ({hit_pct:.1f}% of steps)")
        print(f"  • Cache Misses (API Calls)   : {cache_misses} ({miss_pct:.1f}% of steps)")
        print(f"    - Successful (HTTP 200)    : {successful_api} (saved to cache)")
        if failed_api > 0:
            print(f"    - Failed Calls (NOT Cached): {failed_api}")
            print(f"\n  ⚠️ Failure Breakdown by Reason:")
            for reason, count in failure_reasons.items():
                print(f"    • {reason}: {count} time steps")
        print(f"\n  • OpenSky API Credits Spent  : {cache_misses} credits")
        if credits_remaining is not None:
            print(f"  • Daily API Credits Left     : {credits_remaining}")
        print(f"  • Total Records in BBox Cache: {len(df_bbox):,} rows ({len(queried_ts)} successful time slices)")
        print(f"  • Valid Points for Window    : {len(df_res):,}")
        print(f"  • Unique Aircraft in Window  : {df_res['icao24'].nunique() if not df_res.empty else 0:,}")
        print("=" * 60)
        
        # ----------------------------------------------------------------------
        # Final Report: All Available Time Ranges in the Cache Store
        # ----------------------------------------------------------------------
        timestamps_sorted = sorted(list(queried_ts)) if queried_ts else []
        if not timestamps_sorted and not df_bbox.empty and 'timestamp' in df_bbox.columns:
            timestamps_sorted = sorted(df_bbox['timestamp'].astype(int).unique())
            
        segments = []
        if timestamps_sorted:
            cur_start = timestamps_sorted[0]
            cur_end = timestamps_sorted[0]
            cur_count = 1
            gap_threshold_sec = max(300, time_step_sec * 4)  # Detect gaps > 5 mins
            
            for i in range(1, len(timestamps_sorted)):
                if timestamps_sorted[i] - timestamps_sorted[i - 1] > gap_threshold_sec:
                    segments.append((cur_start, cur_end, cur_count))
                    cur_start = timestamps_sorted[i]
                    cur_end = timestamps_sorted[i]
                    cur_count = 1
                else:
                    cur_end = timestamps_sorted[i]
                    cur_count += 1
            segments.append((cur_start, cur_end, cur_count))
            
        print()
        print("=" * 60)
        print("📅 COMPLETE CACHE STORE TIME RANGE COVERAGE")
        print("=" * 60)
        print(f"  • Bbox Cache File            : {os.path.basename(cache_path)}")
        print(f"  • Total Stored Records       : {len(df_bbox):,} waypoints")
        print(f"  • Total Aircraft Recorded    : {df_bbox['icao24'].nunique() if not df_bbox.empty else 0:,}")
        print(f"  • Distinct Time Segments     : {len(segments)} segment(s)")
        print()
        if segments:
            print("  🕒 Stored Time Windows in Cache:")
            for idx, (s_ts, e_ts, count) in enumerate(segments, 1):
                dt_s = datetime.fromtimestamp(s_ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
                dt_e = datetime.fromtimestamp(e_ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
                span_min = max(0.1, (e_ts - s_ts) / 60.0)
                span_hrs = span_min / 60.0
                span_str = f"{span_hrs:.2f} hrs" if span_hrs >= 1.0 else f"{span_min:.1f} mins"
                print(f"    [{idx}] {dt_s} -> {dt_e} ({span_str}, {count} time slices)")
        else:
            print("  (Cache currently empty)")
        print("=" * 60)
        print()
        
        return df_res


def main():
    parser = argparse.ArgumentParser(description="Fetch and cache OpenSky flight state vectors.")
    parser.add_argument("--lat-min", type=float, default=47.1102, help="Minimum latitude (default: 47.1102)")
    parser.add_argument("--lat-max", type=float, default=47.4855, help="Maximum latitude (default: 47.4855)")
    parser.add_argument("--lon-min", type=float, default=8.3720,  help="Minimum longitude (default: 8.3720)")
    parser.add_argument("--lon-max", type=float, default=9.0478,  help="Maximum longitude (default: 9.0478)")
    parser.add_argument("--lookback-min", type=int, default=50,   help="Lookback minutes from now (default: 50)")
    parser.add_argument("--step-sec", type=int, default=20,       help="Sampling step resolution in seconds (default: 20)")
    parser.add_argument("--active-start-hour", type=int, default=5, help="Earliest operating hour (0-23, default: 5 for 5 AM)")
    parser.add_argument("--active-end-hour", type=int, default=23, help="Latest operating hour (0-23, default: 23 for 11 PM)")
    parser.add_argument("--use-local-tz", action="store_true", default=True, help="Use local timezone for operating hours (default: True)")
    parser.add_argument("--use-utc", dest="use_local_tz", action="store_false", help="Use UTC timezone for operating hours")
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR, help="Cache directory")
    parser.add_argument("--credentials", type=str, default=DEFAULT_CREDS_FILE, help="Path to credentials.json")
    parser.add_argument("--client-id", type=str, default=os.environ.get("OPENSKY_CLIENT_ID", ""), help="OpenSky OAuth2 Client ID (or via OPENSKY_CLIENT_ID env var)")
    parser.add_argument("--client-secret", type=str, default=os.environ.get("OPENSKY_CLIENT_SECRET", ""), help="OpenSky OAuth2 Client Secret (or via OPENSKY_CLIENT_SECRET env var)")
    
    args = parser.parse_args()
    
    _now = datetime.now(timezone.utc)
    # Floor to clean minute boundaries
    start_dt = (_now - timedelta(minutes=args.lookback_min)).replace(second=0, microsecond=0)
    end_dt   = (_now - timedelta(minutes=2)).replace(second=0, microsecond=0)
    
    start_ts = int(start_dt.timestamp())
    end_ts   = int(end_dt.timestamp())
    
    fetcher = OpenSkyStatesFetcher(
        client_id=args.client_id,
        client_secret=args.client_secret,
        cache_dir=args.cache_dir,
        credentials_file=args.credentials
    )
    
    fetcher.fetch_states(
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        start_ts=start_ts,
        end_ts=end_ts,
        time_step_sec=args.step_sec,
        use_cache=True,
        active_start_hour=args.active_start_hour,
        active_end_hour=args.active_end_hour,
        use_local_tz=args.use_local_tz
    )

if __name__ == "__main__":
    main()

