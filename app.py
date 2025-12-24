# app.py
import math
import re
import os
import json
import time
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import requests
import streamlit as st
import folium
from streamlit_folium import st_folium
import streamlit.components.v1 as components
import html as html_lib

# ==============================
# Optional: streamlit-sortables (drag reorder)
# ==============================
try:
    from streamlit_sortables import sort_items  # type: ignore
    SORTABLES_OK = True
except Exception:
    SORTABLES_OK = False
    sort_items = None  # type: ignore

# ==============================
# Optional OpenAI
# ==============================
client = None
OPENAI_OK = False
try:
    from openai import OpenAI  # type: ignore
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip() or (
        st.secrets.get("OPENAI_API_KEY", "").strip() if hasattr(st, "secrets") else ""
    )
    if OPENAI_API_KEY:
        client = OpenAI(api_key=OPENAI_API_KEY)
        OPENAI_OK = True
except Exception:
    client = None
    OPENAI_OK = False

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "").strip() or "gpt-4o-mini"

# ==============================
# Google API Key
# ==============================
API_KEY = os.getenv("GOOGLE_API_KEY", "").strip() or (
    st.secrets.get("GOOGLE_API_KEY", "").strip() if hasattr(st, "secrets") else ""
)

# ==============================
# Constants
# ==============================
FOOD_KEYWORDS = {
    "咖啡 / 咖啡廳": "cafe 咖啡",
    "餐廳（不限）": "restaurant 餐廳",
    "烘焙坊": "bakery 麵包 烘焙",
    "早午餐": "brunch 早午餐",
    "中式料理": "中式 中餐 熱炒",
    "牛肉麵": "牛肉麵",
    "小吃": "小吃 滷肉飯",
    "火鍋": "火鍋",
    "燒肉": "燒肉",
    "燒烤": "燒烤 串燒",
    "拉麵": "拉麵",
    "日式料理": "日式 壽司 丼飯",
    "韓式料理": "韓式 韓式烤肉",
    "義大利料理": "義大利 麵 披薩",
    "美式漢堡": "漢堡 美式",
    "泰式料理": "泰式",
    "越南料理": "越南 河粉",
    "甜點": "甜點 蛋糕",
    "飲料店": "飲料 手搖",
    "素食": "素食 蔬食",
}

ASPECTS = {
    "價格/CP值": ["便宜", "划算", "cp", "CP", "cp值", "CP值", "價格合理", "價位", "貴", "太貴", "平價", "份量"],
    "環境/氛圍": ["環境", "氛圍", "裝潢", "舒適", "安靜", "吵", "明亮", "座位", "插座", "好拍", "乾淨", "不限時", "wifi", "Wi-Fi"],
    "餐點/味道": ["好吃", "美味", "口味", "份量", "食材", "新鮮", "調味", "太鹹", "太甜", "香", "難吃", "招牌"],
    "服務/出餐": ["服務", "態度", "親切", "店員", "出餐", "等很久", "速度", "排隊", "接待"],
}

AUTO_CATS = {
    "讀書": ["安靜", "讀書", "插座", "適合工作", "不限時", "Wi-Fi", "wifi", "筆電", "辦公"],
    "約會": ["氣氛", "浪漫", "好拍", "景觀", "夜景", "裝潢漂亮", "燈光", "生日", "紀念日"],
    "聚會": ["聊天", "聚會", "多人", "包廂", "座位多", "團體", "好停車"],
    "宵夜": ["宵夜", "深夜", "凌晨", "開到", "24小時", "夜貓"],
    "親子": ["親子", "小孩", "兒童", "家庭", "推車"],
    "寵物友善": ["寵物", "狗", "貓", "pet", "友善", "可帶狗"],
    "清真/素": ["素食", "蔬食", "清真", "halal"],
}

TRAVEL_MODES = ["walking", "driving", "transit"]
TRAVEL_MODE_LABEL = {"walking": "步行", "driving": "開車（含路況）", "transit": "大眾運輸"}

PRICE_WORD = {
    1: "小資友善",
    2: "日常舒適",
    3: "進階享受",
    4: "頂級體驗",
}
PRICE_HINT = {
    1: "多數品項偏親民，常吃不心痛",
    2: "一般常見價位，日常聚餐沒壓力",
    3: "價格偏上，適合想吃好一點、品質優先",
    4: "高價位體驗，適合慶祝或特別場合",
}

# ==============================
# UI / CSS
# ==============================
def inject_css() -> None:
    st.markdown(
        """
<style>
:root{
  --bg0:#eef4f7; --bg1:#e9f0f5; --ink:#0f172a;
  --panel: rgba(250,252,255,.76); --panel2: rgba(246,249,252,.66);
  --card: rgba(250,252,255,.92); --card2: rgba(244,248,252,.78);
  --stroke: rgba(15, 23, 42, .12);
  --shadow: 0 14px 50px rgba(2,6,23,.10);
  --shadow2: 0 22px 74px rgba(2,6,23,.14);
  --radius: 20px;
}

html, body, [class*="css"] { font-size: 18px; }
.block-container{ max-width: 1600px !important; padding-top: 1.0rem; padding-bottom: 2.2rem; }

.stApp{
  background:
    radial-gradient(1200px 680px at 14% 10%, rgba(34,197,94,.14), transparent 64%),
    radial-gradient(1100px 680px at 86% 14%, rgba(96,165,250,.16), transparent 62%),
    radial-gradient(950px 540px at 55% 108%, rgba(167,139,250,.10), transparent 62%),
    linear-gradient(180deg, var(--bg0), var(--bg1));
  color: var(--ink);
}

header[data-testid="stHeader"]{ background: transparent; }

section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, var(--panel), var(--panel2));
  border-right: 1px solid var(--stroke);
  backdrop-filter: blur(14px);
}
section[data-testid="stSidebar"] > div { padding-top: 10px; }

@keyframes fadeUp { from{opacity:0; transform: translateY(10px);} to{opacity:1; transform: translateY(0);} }

div[data-testid="stVerticalBlockBorderWrapper"]{
  border: 1px solid rgba(15,23,42,.12);
  background: linear-gradient(180deg, var(--card), var(--card2));
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  backdrop-filter: blur(10px);
  transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
  animation: fadeUp .22s ease both;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover{
  transform: translateY(-2px);
  box-shadow: var(--shadow2);
  border-color: rgba(96,165,250,.22);
}

.stTextInput input, .stTextArea textarea, .stNumberInput input{
  background: rgba(250,252,255,.96) !important;
  border: 1px solid rgba(15,23,42,.18) !important;
  border-radius: 16px !important;
  color: var(--ink) !important;
  min-height: 52px;
  font-size: 18px !important;
}

div[data-baseweb="select"] > div{
  background: rgba(250,252,255,.96) !important;
  border: 1px solid rgba(15,23,42,.18) !important;
  border-radius: 16px !important;
  min-height: 52px;
}
div[data-baseweb="select"] span{ font-size: 18px !important; }

.stButton > button{
  border-radius: 16px !important;
  min-height: 52px;
  font-weight: 900;
  border: 1px solid rgba(15,23,42,.12) !important;
  transition: transform .16s ease, box-shadow .16s ease, filter .16s ease;
  font-size: 18px !important;
}
.stButton > button:hover{
  transform: translateY(-2px);
  box-shadow: 0 14px 30px rgba(2,6,23,.14);
  filter: brightness(1.02);
}
button[kind="primary"]{
  background: linear-gradient(90deg, rgba(34,197,94,.94), rgba(96,165,250,.90)) !important;
  border: none !important;
  color: #06101a !important;
  box-shadow: 0 18px 38px rgba(34,197,94,.16);
}

.badge{
  display:inline-block; padding:10px 14px; border-radius:999px;
  background: rgba(96,165,250,.14); border:1px solid rgba(96,165,250,.22);
  font-size:16px; color: rgba(15,23,42,.88);
  margin-right:10px; margin-bottom:10px;
  font-weight: 900;
}
.badge-ok{ background: rgba(34,197,94,.14); border-color: rgba(34,197,94,.26); }
.badge-warn{ background: rgba(245,158,11,.14); border-color: rgba(245,158,11,.26); }
.badge-danger{ background: rgba(239,68,68,.12); border-color: rgba(239,68,68,.22); }

.small-muted{ color: rgba(15,23,42,.62); font-size: 1.05rem; }
.h1{ font-size: 40px; font-weight: 950; letter-spacing: -0.02em; }
</style>
        """,
        unsafe_allow_html=True,
    )

# ==============================
# Storage
# ==============================
APP_DIR = Path(__file__).resolve().parent
POCKET_FILE = APP_DIR / "pocket_list.json"
HISTORY_FILE = APP_DIR / "search_history.json"
COMPARE_FILE = APP_DIR / "compare_list.json"
TRIP_FILE = APP_DIR / "trip_plan.json"

def _safe_read_json(path: Path, default):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default

def _safe_write_json(path: Path, data) -> None:
    try:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

def load_list_file(path: Path) -> List[Dict[str, Any]]:
    data = _safe_read_json(path, [])
    return data if isinstance(data, list) else []

def save_list_file(path: Path, items: List[Dict[str, Any]]) -> None:
    _safe_write_json(path, items or [])

def load_pocket_list() -> List[Dict[str, Any]]:
    data = load_list_file(POCKET_FILE)
    out: List[Dict[str, Any]] = []
    for x in data:
        if isinstance(x, dict) and isinstance(x.get("place_id"), str) and x["place_id"].strip():
            out.append({
                "place_id": x["place_id"].strip(),
                "name": str(x.get("name", "")).strip(),
                "categories": list(x.get("categories", []) or []),
            })
    seen = set()
    uniq = []
    for it in out:
        if it["place_id"] in seen:
            continue
        seen.add(it["place_id"])
        cats = []
        for c in it.get("categories", []) or []:
            c = str(c).strip()
            if c and c not in cats:
                cats.append(c)
        it["categories"] = cats
        uniq.append(it)
    return uniq

def save_pocket_list(items: List[Dict[str, Any]]) -> None:
    seen = set()
    out = []
    for it in items:
        pid = str(it.get("place_id", "")).strip()
        if not pid or pid in seen:
            continue
        seen.add(pid)
        name = str(it.get("name", "")).strip()
        cats = []
        for c in (it.get("categories", []) or []):
            c = str(c).strip()
            if c and c not in cats:
                cats.append(c)
        out.append({"place_id": pid, "name": name, "categories": cats})
    _safe_write_json(POCKET_FILE, out)

# ==============================
# HTTP helpers
# ==============================
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "Mozilla/5.0 (streamlit-app)"})

def _get_json(url: str, params: Dict[str, Any], timeout: int = 22, retries: int = 2, backoff: float = 0.55) -> Dict[str, Any]:
    last_err = None
    for i in range(retries + 1):
        try:
            r = _SESSION.get(url, params=params, timeout=timeout)
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(backoff * (1.6 ** i) + random.random() * 0.12)
    raise RuntimeError(f"HTTP failed: {last_err}")

# ==============================
# Google APIs
# ==============================
@st.cache_data(show_spinner=False, ttl=900)
def geocode(address: str) -> Optional[Tuple[float, float]]:
    if not API_KEY:
        return None
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    data = _get_json(url, {"address": address, "key": API_KEY, "language": "zh-TW"}, timeout=18)
    if not data.get("results"):
        return None
    loc = data["results"][0]["geometry"]["location"]
    return float(loc["lat"]), float(loc["lng"])

@st.cache_data(show_spinner=False, ttl=900)
def place_from_text(query: str) -> Optional[Dict[str, Any]]:
    if not API_KEY:
        return None
    url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
    params = {
        "input": query,
        "inputtype": "textquery",
        "fields": "place_id,name,formatted_address,geometry",
        "key": API_KEY,
        "language": "zh-TW",
    }
    data = _get_json(url, params, timeout=18)
    candidates = data.get("candidates", [])
    return candidates[0] if candidates else None

@st.cache_data(show_spinner=False, ttl=900)
def nearby_rankby_distance(lat: float, lng: float, keyword: str, open_now: bool) -> List[Dict[str, Any]]:
    if not API_KEY:
        return []
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    base_params: Dict[str, Any] = {
        "location": f"{lat},{lng}",
        "rankby": "distance",
        "keyword": keyword,
        "key": API_KEY,
        "language": "zh-TW",
    }
    if open_now:
        base_params["opennow"] = "true"

    all_results: List[Dict[str, Any]] = []
    page_token = None

    while True:
        if page_token:
            time.sleep(1.7)
            params = {"pagetoken": page_token, "key": API_KEY, "language": "zh-TW"}
        else:
            params = dict(base_params)

        data = _get_json(url, params, timeout=22)
        all_results.extend(data.get("results", []) or [])
        page_token = data.get("next_page_token")
        if not page_token:
            break

    return all_results

@st.cache_data(show_spinner=False, ttl=1200)
def place_details(place_id: str) -> Optional[Dict[str, Any]]:
    if not API_KEY:
        return None
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        "place_id": place_id,
        "fields": ",".join([
            "name", "url", "website", "rating", "user_ratings_total", "reviews",
            "formatted_address", "price_level", "geometry", "opening_hours",
            "formatted_phone_number", "photos"
        ]),
        "key": API_KEY,
        "language": "zh-TW",
    }
    data = _get_json(url, params, timeout=24)
    return data.get("result")

@st.cache_data(show_spinner=False, ttl=3600)
def photo_bytes(photo_reference: str, maxwidth: int = 1600) -> Optional[bytes]:
    if not API_KEY or not photo_reference:
        return None
    url = "https://maps.googleapis.com/maps/api/place/photo"
    try:
        r = _SESSION.get(
            url,
            params={"maxwidth": maxwidth, "photoreference": photo_reference, "key": API_KEY},
            timeout=22,
            allow_redirects=True,
        )
        if r.status_code == 200 and r.content:
            return r.content
    except Exception:
        return None
    return None

@st.cache_data(show_spinner=False, ttl=900)
def directions(origin_lat: float, origin_lng: float, dest_lat: float, dest_lng: float, mode: str = "walking") -> Optional[Dict[str, Any]]:
    if not API_KEY:
        return None
    url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": f"{origin_lat},{origin_lng}",
        "destination": f"{dest_lat},{dest_lng}",
        "mode": mode,
        "key": API_KEY,
        "language": "zh-TW",
        "region": "tw",
    }
    if mode == "driving":
        params["departure_time"] = "now"
        params["traffic_model"] = "best_guess"

    data = _get_json(url, params, timeout=24)
    routes = data.get("routes", [])
    if not routes:
        return None
    route0 = routes[0]
    leg = route0["legs"][0]
    poly = (route0.get("overview_polyline") or {}).get("points")

    dur = leg.get("duration_in_traffic") or leg.get("duration")
    out = {
        "duration_text": (dur or {}).get("text", ""),
        "distance_text": (leg.get("distance") or {}).get("text", ""),
        "steps": leg.get("steps", []) or [],
        "overview_polyline": poly,
        "start_address": leg.get("start_address", ""),
        "end_address": leg.get("end_address", ""),
        "departure_time_text": (leg.get("departure_time") or {}).get("text", ""),
        "arrival_time_text": (leg.get("arrival_time") or {}).get("text", ""),
    }
    return out

# ==============================
# Geometry / Analysis
# ==============================
def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))

def budget_to_price_level(min_twd: int, max_twd: int) -> Tuple[int, int]:
    def to_level(x: int) -> int:
        if x <= 150:
            return 1
        if x <= 350:
            return 2
        if x <= 700:
            return 3
        return 4
    return to_level(min_twd), to_level(max_twd)

def analyze_reviews(reviews: List[Dict[str, Any]]) -> Tuple[List[str], str, str]:
    if not reviews:
        return [], "（沒有可用的文字評論）", ""
    text = "\n".join([rv.get("text", "") for rv in reviews if rv.get("text")])
    if not text.strip():
        return [], "（沒有可用的文字評論）", ""

    hits: Dict[str, int] = {}
    low = text.lower()
    for aspect, kws in ASPECTS.items():
        c = 0
        for kw in kws:
            c += low.count(kw.lower())
        if c > 0:
            hits[aspect] = c

    if not hits:
        return [], "（評論未明確提到價格/環境/餐點/服務關鍵詞）", text

    top = sorted(hits.items(), key=lambda x: x[1], reverse=True)[:3]
    tags = [f"{k}（提及{v}次）" for k, v in top]
    reason = "評價常提到：" + "、".join(tags)
    return tags, reason, text

def suggest_categories_from_text(name: str, review_text: str) -> List[str]:
    base = (name or "") + "\n" + (review_text or "")
    low = base.lower()
    out: List[str] = []
    for cat, kws in AUTO_CATS.items():
        for kw in kws:
            if kw.lower() in low:
                out.append(cat)
                break
    if "咖啡" in base or "cafe" in low:
        out.append("咖啡")
    if "早午餐" in base or "brunch" in low:
        out.append("早午餐")
    seen = set()
    uniq = []
    for c in out:
        c = c.strip()
        if c and c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq[:8]

def score_item(it: Dict[str, Any]) -> float:
    rating = float(it.get("rating", 0) or 0)
    cnt = int(it.get("user_ratings_total", 0) or 0)
    d = float(it.get("distance_km", 9999) or 9999)
    dist_penalty = 2.0 * min(d, 5.0) + 0.5 * max(0.0, d - 5.0)
    return (rating * 22.0) + (math.log1p(cnt) * 3.4) - dist_penalty

# ==============================
# Polyline decode
# ==============================
def decode_polyline(polyline_str: str) -> List[Tuple[float, float]]:
    if not polyline_str:
        return []
    index, lat, lng, coordinates = 0, 0, 0, []
    length = len(polyline_str)

    while index < length:
        shift, result = 0, 0
        while True:
            b = ord(polyline_str[index]) - 63
            index += 1
            result |= (b & 0x1f) << shift
            shift += 5
            if b < 0x20:
                break
        dlat = ~(result >> 1) if (result & 1) else (result >> 1)
        lat += dlat

        shift, result = 0, 0
        while True:
            b = ord(polyline_str[index]) - 63
            index += 1
            result |= (b & 0x1f) << shift
            shift += 5
            if b < 0x20:
                break
        dlng = ~(result >> 1) if (result & 1) else (result >> 1)
        lng += dlng

        coordinates.append((lat / 1e5, lng / 1e5))
    return coordinates

# ==============================
# Photo carousel (Google-style hover arrows + auto-play)
# ==============================
def render_photo_carousel_ui(photo_refs: List[str], key: str, maxwidth: int = 1600, height: int = 360, interval_ms: int = 2400) -> None:
    refs = [r for r in (photo_refs or []) if r]
    if not refs:
        st.info("此店沒有可用照片（可直接開 Google Maps 看照片）")
        return

    urls = [
        f"https://maps.googleapis.com/maps/api/place/photo?maxwidth={maxwidth}&photoreference={r}&key={API_KEY}"
        for r in refs[:14]
    ]
    safe_urls = json.dumps(urls, ensure_ascii=False)

    carousel_html_str = f"""
    <div class="gc-wrap" id="wrap-{key}">
      <div class="gc-stage" id="stage-{key}">
        <img id="img-{key}" src="" />
        <button class="gc-nav gc-prev" id="prev-{key}" aria-label="prev">❮</button>
        <button class="gc-nav gc-next" id="next-{key}" aria-label="next">❯</button>
      </div>
      <div class="gc-dots" id="dots-{key}"></div>
    </div>

    <style>
      .gc-wrap{{ width:100%; }}
      .gc-stage{{
        position:relative; width:100%;
        height:{height}px; overflow:hidden;
        border-radius:16px;
        background: rgba(0,0,0,.04);
      }}
      .gc-stage img{{
        width:100%; height:100%;
        object-fit:cover;
        display:block;
      }}
      .gc-nav{{
        position:absolute; top:50%;
        transform:translateY(-50%);
        width:48px; height:48px;
        border-radius:999px;
        border:1px solid rgba(255,255,255,.38);
        background: rgba(15,23,42,.28);
        color:#fff;
        font-size:22px;
        cursor:pointer;
        opacity:0;
        transition: opacity .18s ease, background .18s ease;
        backdrop-filter: blur(8px);
      }}
      .gc-stage:hover .gc-nav{{ opacity:1; }}
      .gc-nav:hover{{ background: rgba(15,23,42,.42); }}
      .gc-prev{{ left:10px; }}
      .gc-next{{ right:10px; }}
      .gc-dots{{
        display:flex; gap:8px; justify-content:center;
        margin-top:10px; flex-wrap:wrap;
      }}
      .gc-dot{{
        width:9px; height:9px; border-radius:999px;
        background: rgba(15,23,42,.22);
        cursor:pointer;
      }}
      .gc-dot.active{{ background: rgba(34,197,94,.72); }}
    </style>

    <script>
      (function(){{
        const urls = {safe_urls};
        const img = document.getElementById("img-{key}");
        const prev = document.getElementById("prev-{key}");
        const next = document.getElementById("next-{key}");
        const dots = document.getElementById("dots-{key}");
        let i = 0;
        let timer = null;

        function renderDots(){{
          dots.innerHTML = "";
          urls.forEach((_, idx) => {{
            const d = document.createElement("div");
            d.className = "gc-dot" + (idx===i ? " active" : "");
            d.addEventListener("click", () => {{
              i = idx;
              show();
              reset();
            }});
            dots.appendChild(d);
          }});
        }}

        function show(){{
          img.src = urls[i];
          Array.from(dots.children).forEach((d, idx) => {{
            d.className = "gc-dot" + (idx===i ? " active" : "");
          }});
        }}

        function step(dir){{
          i = (i + dir + urls.length) % urls.length;
          show();
        }}

        function reset(){{
          if(timer) clearInterval(timer);
          timer = setInterval(() => step(1), {interval_ms});
        }}

        prev.addEventListener("click", () => {{ step(-1); reset(); }});
        next.addEventListener("click", () => {{ step(1); reset(); }});

        renderDots();
        show();
        reset();
      }})();
    </script>
    """
    components.html(carousel_html_str, height=height + 110, scrolling=False)

# ==============================
# AI
# ==============================
def _local_ai(prompt: str, pool: List[Dict[str, Any]]) -> str:
    if not pool:
        return "候選池是空的：先去「推薦」搜尋一次，或去「口袋名單」收藏一些店再切到 AI。"

    ui = (prompt or "").strip()
    want_top5 = any(k in ui for k in ["5家", "五家", "top5", "前五"])
    want_near = any(k in ui for k in ["近", "附近", "走路", "不要太遠"])
    want_quiet = any(k in ui for k in ["安靜", "讀書", "工作", "插座", "不限時", "wifi", "Wi-Fi", "筆電"])
    want_cp = any(k in ui.lower() for k in ["cp", "便宜", "划算", "平價", "不貴"])
    use_compare = any(k in ui for k in ["比較清單", "比較"])

    if use_compare and st.session_state.get("compare_ids"):
        picked_ids = list(st.session_state.compare_ids)
        pool2 = [c for c in pool if c.get("place_id") in picked_ids]
    else:
        pool2 = pool

    scored = []
    for c in pool2:
        s = float(c.get("score", 0) or 0)
        d = float(c.get("distance_km", 999) or 999)
        reason = (c.get("reason", "") or "")
        if want_near:
            s += max(0.0, 3.0 - d) * 3.2
        if want_quiet and ("環境/氛圍" in reason):
            s += 6.5
        if want_cp and ("價格/CP值" in reason):
            s += 6.0
        scored.append((s, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    k = 5 if want_top5 else 8
    pick = [c for _, c in scored[:k]]

    lines = []
    lines.append("推薦：")
    for i, c in enumerate(pick, 1):
        lines.append(f"{i}. {c.get('name','未命名')}（place_id：{c.get('place_id','')}）｜⭐{c.get('rating',0)}｜{float(c.get('distance_km',0) or 0):.2f}km｜評論{c.get('user_ratings_total',0)}")
    if pick:
        lines.append(f"最適合你：{pick[0].get('name','')}（place_id：{pick[0].get('place_id','')}）— 以距離、評分、評論量綜合下來分數最高。")
    return "\n".join(lines)

def openai_chat(prompt: str, pool: List[Dict[str, Any]], history: List[Dict[str, str]], compare_ids: List[str]) -> str:
    if not (client and OPENAI_OK):
        return _local_ai(prompt, pool)

    compact_pool = []
    for c in pool[:180]:
        compact_pool.append({
            "place_id": c.get("place_id", ""),
            "name": c.get("name", ""),
            "rating": c.get("rating", 0),
            "distance_km": c.get("distance_km", 999),
            "user_ratings_total": c.get("user_ratings_total", 0),
            "score": c.get("score", 0),
            "review_reason": c.get("reason", ""),
        })

    sys = (
        "你是台灣在地餐廳/咖啡廳推薦顧問，回覆用繁體中文。"
        "你只能依據候選清單做篩選/比較/推薦，不能編造不存在店家，而且回答一定要給店名。"
        "如果 compare_ids 非空，且使用者提到『比較』或『比較清單』，你必須以 compare_ids 內店家為主。"
        "輸出格式固定：\n"
        "A) 需求摘要（1-2行）\n"
        "B) 每一家店的適配點評（對使用者情境，至少4句；每家必須含 place_id）\n"
        "C) 你的比較分析（把使用者需求逐條對照，點出哪家最適合與為什麼）\n"
        "D) 最終只選一家（加粗店名），並再補2句『為什麼就是它』\n"
        "E) 如果需要可追問最多2題（可選）"
    )

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": sys},
                *history[-12:],
                {"role": "user", "content": f"compare_ids={compare_ids}\n候選清單(JSON)={compact_pool}\n\n我的需求：{prompt}"},
            ],
            temperature=0.55,
        )
        txt = (resp.choices[0].message.content or "").strip()
        return txt or _local_ai(prompt, pool)
    except Exception:
        return _local_ai(prompt, pool)

def extract_place_ids_from_text(txt: str) -> List[str]:
    if not txt:
        return []
    pids = re.findall(r"place_id\s*[:=：]\s*([A-Za-z0-9_\-]+)", txt)
    if not pids:
        pids = re.findall(r"\(([A-Za-z0-9_\-]{10,})\)", txt)
    out = []
    seen = set()
    for pid in pids:
        pid = pid.strip()
        if pid and pid not in seen:
            seen.add(pid)
            out.append(pid)
    return out[:20]

# ==============================
# Session init
# ==============================
def ss_init():
    if "search_done" not in st.session_state: st.session_state.search_done = False
    if "results" not in st.session_state: st.session_state.results = []
    if "filtered" not in st.session_state: st.session_state.filtered = []
    if "origin" not in st.session_state: st.session_state.origin = None

    if "page" not in st.session_state: st.session_state.page = 1
    if "per_page" not in st.session_state: st.session_state.per_page = 20
    if "selected_pid" not in st.session_state: st.session_state.selected_pid = None

    if "pocket_list" not in st.session_state: st.session_state.pocket_list = load_pocket_list()
    if "history" not in st.session_state: st.session_state.history = load_list_file(HISTORY_FILE)

    if "compare_ids" not in st.session_state:
        st.session_state.compare_ids = [x.get("place_id") for x in load_list_file(COMPARE_FILE) if x.get("place_id")]

    if "trip_items" not in st.session_state:
        raw = load_list_file(TRIP_FILE)
        out = []
        for x in raw:
            if isinstance(x, dict) and x.get("place_id"):
                out.append({
                    "place_id": str(x.get("place_id")).strip(),
                    "name": str(x.get("name", "")).strip(),
                    "stay_min": int(x.get("stay_min", 45) or 45),
                })
        st.session_state.trip_items = out

    if "route_mode" not in st.session_state: st.session_state.route_mode = "driving"

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "你可以直接說：『只留安靜可讀書＋近』／『用比較清單給結論』／『挑最適合聚會的5家』"}
        ]
    if "ai_source" not in st.session_state: st.session_state.ai_source = "本次搜尋結果"
    if "flash" not in st.session_state: st.session_state.flash = ""
    if "pending_ai_prompt" not in st.session_state: st.session_state.pending_ai_prompt = ""

def save_compare_ids():
    save_list_file(COMPARE_FILE, [{"place_id": pid} for pid in st.session_state.compare_ids])

def save_trip_items():
    save_list_file(TRIP_FILE, st.session_state.trip_items)

# ==============================
# Pocket helpers
# ==============================
def pocket_has(pid: str) -> bool:
    return any(x["place_id"] == pid for x in st.session_state.pocket_list)

def pocket_get(pid: str) -> Optional[Dict[str, Any]]:
    for x in st.session_state.pocket_list:
        if x["place_id"] == pid:
            return x
    return None

def all_categories() -> List[str]:
    s = set()
    for it in st.session_state.pocket_list:
        for c in it.get("categories", []) or []:
            s.add(str(c))
    return sorted(s)

def pocket_set_categories(pid: str, cats: List[str]) -> None:
    it = pocket_get(pid)
    if not it:
        return
    cleaned = []
    for c in cats:
        c = str(c).strip()
        if c and c not in cleaned:
            cleaned.append(c)
    it["categories"] = cleaned
    save_pocket_list(st.session_state.pocket_list)

def pocket_add(pid: str, name: str = "") -> None:
    if not pid:
        return
    if pocket_has(pid):
        st.session_state.flash = "ℹ️ 已在口袋名單內"
        return

    cats: List[str] = []
    if API_KEY:
        d = place_details(pid) or {}
        reviews = d.get("reviews", []) or []
        _, _, raw = analyze_reviews(reviews)
        cats = suggest_categories_from_text(d.get("name", name) or name, raw)

    st.session_state.pocket_list.append({"place_id": pid, "name": name, "categories": cats})
    save_pocket_list(st.session_state.pocket_list)
    st.session_state.flash = f"✅ 已收藏：{name or pid}" + (f"（自動分類：{'、'.join(cats)}）" if cats else "")

def pocket_remove(pid: str) -> None:
    st.session_state.pocket_list = [x for x in st.session_state.pocket_list if x["place_id"] != pid]
    save_pocket_list(st.session_state.pocket_list)
    st.session_state.flash = "🗑️ 已移除收藏"

# ==============================
# History helpers
# ==============================
def save_history(items: List[Dict[str, Any]]) -> None:
    save_list_file(HISTORY_FILE, (items or [])[:40])

def push_history(entry: Dict[str, Any]) -> None:
    sig = entry.get("sig")
    hist = [x for x in st.session_state.history if x.get("sig") != sig]
    hist.insert(0, entry)
    st.session_state.history = hist[:40]
    save_history(st.session_state.history)

# ==============================
# Search
# ==============================
def do_search(
    location_text_: str,
    shop_query_: str,
    food_types_: List[str],
    min_budget_: int,
    max_budget_: int,
    min_rating_: float,
    open_now_: bool,
) -> None:
    if not API_KEY:
        st.error("你沒有設定 Google API Key（GOOGLE_API_KEY），無法搜尋。")
        return

    origin = geocode(location_text_)
    if not origin:
        st.error("找不到這個地點，請換更明確的地標/地址")
        return

    lat, lng = origin
    st.session_state.origin = (lat, lng)

    if (shop_query_ or "").strip():
        fp = place_from_text(shop_query_.strip())
        if not fp:
            st.error("找不到這家店，試著輸入更完整的店名/地址。")
            return

        pid = fp["place_id"]
        geo = (fp.get("geometry", {}) or {}).get("location")
        dest_lat = float(geo["lat"]) if geo else lat
        dest_lng = float(geo["lng"]) if geo else lng
        dkm = haversine_km(lat, lng, dest_lat, dest_lng)

        base = {
            "place_id": pid,
            "name": fp.get("name") or shop_query_.strip(),
            "rating": 0.0,
            "user_ratings_total": 0,
            "distance_km": float(dkm),
            "price_level": None,
            "lat": dest_lat,
            "lng": dest_lng,
        }
        base["score"] = score_item(base)

        st.session_state.results = [base]
        st.session_state.search_done = True
        st.session_state.page = 1
        st.session_state.selected_pid = pid

        sig = ("DIRECT", location_text_, shop_query_.strip())
        push_history({
            "sig": sig,
            "title": f"搜尋：{shop_query_.strip()} @ {location_text_}",
            "payload": {"location_text": location_text_, "shop_query": shop_query_.strip()},
        })
        return

    if not food_types_:
        st.warning("請至少選一種餐飲類型")
        return
    if min_budget_ > max_budget_:
        st.warning("最低預算不能高於最高預算")
        return

    keywords = [FOOD_KEYWORDS[x] for x in food_types_]
    candidates: List[Dict[str, Any]] = []
    for kw in keywords:
        candidates.extend(nearby_rankby_distance(lat, lng, kw, open_now=open_now_))

    seen = set()
    enriched: List[Dict[str, Any]] = []
    min_r = float(min_rating_)

    for p in candidates:
        pid = p.get("place_id")
        if not pid or pid in seen:
            continue
        seen.add(pid)

        rating = float(p.get("rating", 0) or 0)
        if rating < min_r:
            continue

        geo = (p.get("geometry", {}) or {}).get("location")
        if not geo:
            continue
        plat = float(geo["lat"])
        plng = float(geo["lng"])
        dkm = haversine_km(lat, lng, plat, plng)

        enriched.append({
            "place_id": pid,
            "name": p.get("name", ""),
            "rating": rating,
            "user_ratings_total": int(p.get("user_ratings_total", 0) or 0),
            "distance_km": dkm,
            "price_level": p.get("price_level", None),
            "lat": plat,
            "lng": plng,
        })

    min_pl, max_pl = budget_to_price_level(int(min_budget_), int(max_budget_))
    after_budget = []
    for it in enriched:
        pl = it["price_level"]
        if pl is None or (min_pl <= int(pl) <= max_pl):
            it["score"] = score_item(it)
            after_budget.append(it)

    after_budget.sort(key=lambda x: x.get("score", 0), reverse=True)

    st.session_state.results = after_budget
    st.session_state.search_done = True
    st.session_state.page = 1
    st.session_state.selected_pid = (after_budget[0]["place_id"] if after_budget else None)

    sig = ("NEARBY", location_text_, tuple(sorted(food_types_)), int(min_budget_), int(max_budget_), float(min_rating_), bool(open_now_))
    push_history({
        "sig": sig,
        "title": f"{location_text_}｜{','.join(food_types_[:2])}{'…' if len(food_types_)>2 else ''}｜⭐{min_rating_}+",
        "payload": {
            "location_text": location_text_,
            "shop_query": "",
            "food_types": food_types_,
            "min_budget": int(min_budget_),
            "max_budget": int(max_budget_),
            "min_rating": float(min_rating_),
            "open_now": bool(open_now_),
        },
    })

# ==============================
# Map
# ==============================
def _extract_place_id_from_popup(popup_html: Any) -> Optional[str]:
    if not popup_html:
        return None
    s = str(popup_html)
    m = re.search(r"place_id\s*[:：]\s*([A-Za-z0-9_\-]+)", s)
    return m.group(1).strip() if m else None

def _marker_style(pid: str, selected_pid: Optional[str]) -> Tuple[str, str]:
    if selected_pid and pid == selected_pid:
        return ("purple", "star")
    if any(x.get("place_id") == pid for x in st.session_state.trip_items):
        return ("red", "flag")
    if pid in st.session_state.compare_ids:
        return ("blue", "ok-sign")
    if pocket_has(pid):
        return ("orange", "heart")
    return ("green", "cutlery")

def build_map(items: List[Dict[str, Any]], origin: Optional[Tuple[float, float]], selected_pid: Optional[str]) -> Dict[str, Any]:
    if not origin:
        st.info("沒有起點座標，無法顯示地圖")
        return {}

    m = folium.Map(location=[origin[0], origin[1]], zoom_start=14, control_scale=True)
    folium.Marker(
        [origin[0], origin[1]],
        tooltip="你的位置",
        icon=folium.Icon(color="darkblue", icon="user"),
    ).add_to(m)

    for it in items:
        pid = it["place_id"]
        lat = it.get("lat")
        lng = it.get("lng")
        if lat is None or lng is None:
            continue

        name = it.get("name", "") or "未命名"
        color, icon = _marker_style(pid, selected_pid)

        badges = []
        if pocket_has(pid): badges.append("口袋")
        if pid in st.session_state.compare_ids: badges.append("比較")
        if any(x.get("place_id") == pid for x in st.session_state.trip_items): badges.append("行程")
        btxt = ("｜" + "、".join(badges)) if badges else ""

        pl = it.get("price_level", None)
        pl_txt = ""
        if pl is not None:
            pl_txt = f"｜價位 {PRICE_WORD.get(int(pl), str(pl))}"

        popup_html = f"""
        <div style="min-width:280px">
          <b>{html_lib.escape(name)}</b>{html_lib.escape(btxt)}<br/>
          ⭐ {it.get("rating",0)}｜評論 {it.get("user_ratings_total",0)}{html_lib.escape(pl_txt)}<br/>
          距離 {float(it.get("distance_km",0) or 0):.2f} km<br/>
          <span style="font-size:12px;color:#666">place_id：{pid}</span>
        </div>
        """

        folium.Marker(
            [float(lat), float(lng)],
            tooltip=name,
            popup=folium.Popup(popup_html, max_width=520),
            icon=folium.Icon(color=color, icon=icon),
        ).add_to(m)

    out = st_folium(m, width=None, height=600)
    return out or {}

def build_route_map(origin: Tuple[float, float], dest: Tuple[float, float], polyline_points: List[Tuple[float, float]]) -> folium.Map:
    center = [(origin[0] + dest[0]) / 2, (origin[1] + dest[1]) / 2]
    m = folium.Map(location=center, zoom_start=13, control_scale=True)
    folium.Marker([origin[0], origin[1]], tooltip="起點", icon=folium.Icon(color="darkblue", icon="user")).add_to(m)
    folium.Marker([dest[0], dest[1]], tooltip="目的地", icon=folium.Icon(color="red", icon="flag")).add_to(m)
    if polyline_points:
        folium.PolyLine(polyline_points, weight=6, opacity=0.9).add_to(m)
    return m

# ==============================
# Drawer
# ==============================
def render_drawer(pid: Optional[str], origin: Optional[Tuple[float, float]]):
    with st.container(border=True):
        st.markdown("## 🧾 店家櫥窗")
        if not pid:
            st.caption("👉 在地圖點店家，或在清單按「查看」")
            return

        d = place_details(pid) or {}
        name = d.get("name", pid)
        url = d.get("url") or f"https://www.google.com/maps/place/?q=place_id:{pid}"

        st.markdown(f"### [{name}]({url})")
        addr = d.get("formatted_address", "")
        if addr:
            st.caption(f"📍 {addr}")

        t1, t2, t3 = st.tabs(["總覽", "導航", "評論"])

        with t1:
            photos = d.get("photos", []) or []
            refs = [p.get("photo_reference") for p in photos if p.get("photo_reference")][:14]
            if refs and API_KEY:
                render_photo_carousel_ui(refs, key=f"{pid}", maxwidth=1600, height=380, interval_ms=2300)
            else:
                st.info("此店沒有可用照片（可直接開 Google Maps 看照片）")

            pl = d.get("price_level", None)
            if pl is not None:
                pword = PRICE_WORD.get(int(pl), f"等級{pl}")
                phint = PRICE_HINT.get(int(pl), "")
                st.markdown(f'<span class="badge">💰 價位：{html_lib.escape(pword)}</span>', unsafe_allow_html=True)
                if phint:
                    st.caption(phint)

            oh = d.get("opening_hours", {})
            if isinstance(oh, dict):
                flag = oh.get("open_now", None)
                if flag is True:
                    st.markdown('<span class="badge badge-ok">🟢 目前營業中</span>', unsafe_allow_html=True)
                elif flag is False:
                    st.markdown('<span class="badge badge-warn">🔴 目前未營業</span>', unsafe_allow_html=True)

            phone = d.get("formatted_phone_number")
            website = d.get("website")
            if phone:
                st.markdown(f'<span class="badge">📞 {html_lib.escape(str(phone))}</span>', unsafe_allow_html=True)
            if website:
                st.markdown(f"🌐 官網：{website}")

            a1, a2, a3 = st.columns(3)
            with a1:
                if pocket_has(pid):
                    if st.button("🗑️ 移除收藏", key=f"drawer_del_{pid}", use_container_width=True):
                        pocket_remove(pid)
                        st.rerun()
                else:
                    if st.button("➕ 收藏（自動分類）", key=f"drawer_add_{pid}", use_container_width=True):
                        pocket_add(pid, name)
                        st.rerun()

            with a2:
                in_cmp = pid in st.session_state.compare_ids
                if st.button("✅ 已在比較" if in_cmp else "➕ 加入比較", key=f"drawer_cmp_{pid}", use_container_width=True):
                    if not in_cmp:
                        st.session_state.compare_ids.append(pid)
                        save_compare_ids()
                    st.rerun()

            with a3:
                in_trip = any(x.get("place_id") == pid for x in st.session_state.trip_items)
                if st.button("✅ 已在行程" if in_trip else "🧭 加入行程", key=f"drawer_trip_{pid}", use_container_width=True):
                    if not in_trip:
                        st.session_state.trip_items.append({"place_id": pid, "name": name, "stay_min": 45})
                        save_trip_items()
                    st.rerun()

        with t2:
            geo = (d.get("geometry", {}) or {}).get("location")
            if origin and geo:
                dest = (float(geo["lat"]), float(geo["lng"]))

                mode = st.selectbox(
                    "路線模式",
                    TRAVEL_MODES,
                    index=TRAVEL_MODES.index(st.session_state.route_mode) if st.session_state.route_mode in TRAVEL_MODES else 1,
                    format_func=lambda x: TRAVEL_MODE_LABEL.get(x, x),
                    key=f"drawer_route_mode_{pid}",
                )
                st.session_state.route_mode = mode

                info = directions(origin[0], origin[1], dest[0], dest[1], mode=mode)
                if info:
                    c1, c2 = st.columns(2)
                    c1.metric("距離", info["distance_text"])
                    c2.metric("時間", info["duration_text"])

                    if mode == "transit":
                        dep = info.get("departure_time_text", "")
                        arr = info.get("arrival_time_text", "")
                        if dep or arr:
                            st.info(f"🚌 出發：{dep}　｜　到達：{arr}")

                        with st.expander("大眾運輸明細", expanded=True):
                            for s in (info.get("steps") or []):
                                tm = (s.get("travel_mode") or "").upper()

                                if tm == "WALKING":
                                    inst = re.sub(r"<[^>]+>", "", s.get("html_instructions", "") or "")
                                    dist = (s.get("distance", {}) or {}).get("text", "")
                                    dur2 = (s.get("duration", {}) or {}).get("text", "")
                                    if inst.strip():
                                        st.write(f"🚶 {inst}（{dist}｜{dur2}）")

                                elif tm == "TRANSIT":
                                    td = s.get("transit_details") or {}
                                    line = (td.get("line") or {})
                                    vehicle = ((line.get("vehicle") or {}).get("name")) or "交通工具"
                                    short = line.get("short_name") or line.get("name") or ""
                                    headsign = td.get("headsign") or ""
                                    stops = td.get("num_stops")
                                    dep_stop = ((td.get("departure_stop") or {}).get("name")) or ""
                                    arr_stop = ((td.get("arrival_stop") or {}).get("name")) or ""
                                    dep_t = ((td.get("departure_time") or {}).get("text")) or ""
                                    arr_t = ((td.get("arrival_time") or {}).get("text")) or ""
                                    st.write(f"🚌 {vehicle} {short} → {headsign}｜{dep_stop}({dep_t}) → {arr_stop}({arr_t})｜{stops}站")

                    poly = decode_polyline(info.get("overview_polyline", "") or "")
                    m = build_route_map(origin, dest, poly)
                    st_folium(m, height=420)

                    with st.expander("逐步導航", expanded=False):
                        steps = info.get("steps", []) or []
                        for s in steps[:40]:
                            inst = re.sub(r"<[^>]+>", "", s.get("html_instructions", "") or "")
                            dist = (s.get("distance", {}) or {}).get("text", "")
                            dur = (s.get("duration", {}) or {}).get("text", "")
                            if inst.strip():
                                st.write(f"- {inst}（{dist}｜{dur}）")

                    maps_nav = f"https://www.google.com/maps/dir/?api=1&origin={origin[0]},{origin[1]}&destination={dest[0]},{dest[1]}&travelmode={mode}"
                    st.markdown(f"➡️ [用 Google Maps 開啟即時導航]({maps_nav})")
                else:
                    st.warning("Directions 查不到路線（可直接開 Google Maps）")
                    st.markdown(f"➡️ [打開 Google Maps]({url})")
            else:
                st.caption("（缺少起點或店家座標，無法導航）")

        with t3:
            reviews = d.get("reviews", []) or []
            tags, reason, _raw = analyze_reviews(reviews)
            st.write("**評論重點：**")
            st.write(reason)
            if tags:
                st.markdown(" ".join([f'<span class="badge">{html_lib.escape(t)}</span>' for t in tags]), unsafe_allow_html=True)
            if reviews:
                with st.expander("看幾則 Google 評論（節錄）", expanded=False):
                    for rv in reviews[:10]:
                        txt = (rv.get("text", "") or "").strip()
                        star = rv.get("rating", "")
                        if txt:
                            st.write(f"⭐ {star}：{txt[:520]}{'…' if len(txt)>520 else ''}")

# ==============================
# Compare panel
# ==============================
def render_compare_manager(lookup: Dict[str, Dict[str, Any]], context_key: str = "cmp"):
    ids = list(st.session_state.compare_ids)
    if not ids:
        st.caption("（目前比較清單是空的）")
        return

    rows = []
    for pid in ids[:20]:
        it = lookup.get(pid) or (place_details(pid) or {})
        pl = it.get("price_level", None)
        rows.append({
            "店名": it.get("name", pid),
            "place_id": pid,
            "評分": it.get("rating", 0),
            "評論數": it.get("user_ratings_total", 0),
            "價位": PRICE_WORD.get(int(pl), "未知") if pl is not None else "未知",
        })

    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.markdown("#### 管理比較清單")
    for pid in ids[:20]:
        cols = st.columns([7, 3])
        name = (lookup.get(pid) or {}).get("name", pid)
        cols[0].write(f"- {name}（{pid}）")
        if cols[1].button("移除", key=f"{context_key}_rm_{pid}", use_container_width=True):
            st.session_state.compare_ids = [x for x in st.session_state.compare_ids if x != pid]
            save_compare_ids()
            st.rerun()

    c1, c2 = st.columns(2)
    if c1.button("🧹 清空比較", key=f"{context_key}_clear", use_container_width=True):
        st.session_state.compare_ids = []
        save_compare_ids()
        st.rerun()
    if c2.button("🤖 以比較清單下結論", key=f"{context_key}_ask_ai", use_container_width=True):
        st.session_state.pending_ai_prompt = "請以『比較清單』優先，先逐家點評（每家4-5句），再做需求對照比較，最後只選一家最適合我並強調原因。"
        st.rerun()

# ==============================
# Trip
# ==============================
def _parse_time(s: str) -> Optional[datetime]:
    try:
        return datetime.strptime(s, "%H:%M")
    except Exception:
        return None

def _fmt_time(dt: datetime) -> str:
    return dt.strftime("%H:%M")

def _extract_minutes_from_duration_text(t: str) -> int:
    if not t:
        return 0
    mins = 0
    m1 = re.search(r"(\d+)\s*小時", t)
    m2 = re.search(r"(\d+)\s*分鐘", t)
    if m1: mins += int(m1.group(1)) * 60
    if m2: mins += int(m2.group(1))
    return mins

def plan_trip(origin: Optional[Tuple[float, float]], lookup: Dict[str, Dict[str, Any]]):
    st.markdown("## 🧭 行程規劃")
    if not origin:
        st.info("先做一次搜尋（建立起點座標）")
        return

    if not st.session_state.trip_items:
        st.caption("（尚未加入任何行程點：在櫥窗或清單按『🧭 加入行程』）")
        return

    st.markdown("### 行程順序（拖拉排序）")
    if SORTABLES_OK:
        labels = []
        for it in st.session_state.trip_items:
            pid = it.get("place_id", "")
            name = (lookup.get(pid) or {}).get("name") or it.get("name") or pid
            labels.append(f"{name}｜{pid}")
        new_labels = sort_items(labels, direction="vertical")
        new_trip = []
        for lab in new_labels:
            pid = lab.split("｜")[-1].strip()
            old = next((x for x in st.session_state.trip_items if x.get("place_id") == pid), None)
            if old:
                new_trip.append(old)
        if new_trip and [x.get("place_id") for x in new_trip] != [x.get("place_id") for x in st.session_state.trip_items]:
            st.session_state.trip_items = new_trip
            save_trip_items()
    else:
        st.warning("目前環境未安裝 streamlit-sortables，拖拉排序無法啟用。請先：pip install streamlit-sortables")

    st.markdown("### 行程清單（可改停留時間）")
    for it in list(st.session_state.trip_items):
        pid = it["place_id"]
        name = (lookup.get(pid) or {}).get("name") or it.get("name") or pid

        row = st.columns([6, 2, 2])
        row[0].write(f"**{name}**  \n`{pid}`")
        stay = row[1].number_input("停留(分)", min_value=0, max_value=600, value=int(it.get("stay_min", 45)), step=5, key=f"trip_stay_{pid}")
        if row[2].button("移除", key=f"trip_rm_{pid}", use_container_width=True):
            st.session_state.trip_items = [x for x in st.session_state.trip_items if x.get("place_id") != pid]
            save_trip_items()
            st.rerun()

        it["name"] = name
        it["stay_min"] = int(stay)

    c1, c2 = st.columns(2)
    if c1.button("🧹 清空行程", key="trip_clear_all", use_container_width=True):
        st.session_state.trip_items = []
        save_trip_items()
        st.rerun()

    travelmode = c2.selectbox("移動方式", TRAVEL_MODES, index=1, format_func=lambda x: TRAVEL_MODE_LABEL.get(x, x), key="trip_mode")
    start_time_str = st.text_input("開始時間（HH:MM）", value="18:00", key="trip_start_time")

    pts: List[Dict[str, Any]] = []
    for it in st.session_state.trip_items:
        pid = str(it.get("place_id", "")).strip()
        if not pid:
            continue
        base = lookup.get(pid) or {}
        lat = base.get("lat")
        lng = base.get("lng")
        name = it.get("name") or base.get("name") or pid
        if lat is None or lng is None:
            d = place_details(pid) or {}
            geo = (d.get("geometry", {}) or {}).get("location")
            if geo:
                lat, lng = float(geo["lat"]), float(geo["lng"])
        if lat is not None and lng is not None:
            pts.append({
                "place_id": pid,
                "name": name,
                "lat": float(lat),
                "lng": float(lng),
                "stay_min": int(it.get("stay_min", 45) or 45)
            })

    if not pts:
        st.warning("行程點缺少座標（請先讓店家 details 可取得）")
        return

    base_time = _parse_time(start_time_str) or datetime.strptime("18:00", "%H:%M")

    @st.cache_data(show_spinner=False, ttl=900)
    def _cached_leg_minutes(a_lat: float, a_lng: float, b_lat: float, b_lng: float, mode: str) -> int:
        info = directions(a_lat, a_lng, b_lat, b_lng, mode=mode)
        if not info:
            d = haversine_km(a_lat, a_lng, b_lat, b_lng)
            speed_kmh = 18 if mode == "driving" else (5 if mode == "walking" else 12)
            return int(max(3, (d / speed_kmh) * 60))
        mins = _extract_minutes_from_duration_text(info.get("duration_text", ""))
        return int(max(1, mins))

    st.markdown("### 時間軸")
    tcur = base_time
    cur_lat, cur_lng = origin[0], origin[1]
    timeline_rows = []

    for i, p in enumerate(pts, 1):
        move_m = _cached_leg_minutes(cur_lat, cur_lng, p["lat"], p["lng"], travelmode)
        arrive = tcur + timedelta(minutes=move_m)
        leave = arrive + timedelta(minutes=int(p.get("stay_min", 45)))
        timeline_rows.append({
            "順序": i,
            "店": p["name"],
            "place_id": p["place_id"],
            "移動(分)": move_m,
            "到達": _fmt_time(arrive),
            "停留(分)": int(p.get("stay_min", 45)),
            "離開": _fmt_time(leave),
        })
        tcur = leave
        cur_lat, cur_lng = p["lat"], p["lng"]

    st.dataframe(timeline_rows, use_container_width=True, hide_index=True)

    m = folium.Map(location=[origin[0], origin[1]], zoom_start=13, control_scale=True)
    folium.Marker([origin[0], origin[1]], tooltip="起點", icon=folium.Icon(color="darkblue", icon="user")).add_to(m)
    for i, p in enumerate(pts, 1):
        folium.Marker([p["lat"], p["lng"]], tooltip=f"{i}. {p['name']}", icon=folium.Icon(color="red", icon="flag")).add_to(m)
    folium.PolyLine([(origin[0], origin[1])] + [(p["lat"], p["lng"]) for p in pts], weight=5, opacity=0.75).add_to(m)
    st_folium(m, height=460)

# ==============================
# App start
# ==============================
st.set_page_config(page_title="別當我", layout="wide")
inject_css()
ss_init()

st.markdown('<div class="h1">📍 別當我</div>', unsafe_allow_html=True)
st.markdown('<div class="small-muted">輸入店名＝直接查；不輸入＝附近推薦。支援：口袋名單/比較/AI/導航/行程。</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🔎 搜尋")
    with st.form("search_form", clear_on_submit=False):
        location_text = st.text_input("你目前在哪？（地標/地址）", "勤美術館", key="loc_text")
        shop_query = st.text_input("店名 / 關鍵字（選填：填了就直接查店）", placeholder="例：春水堂 勤美 / 鼎泰豐 台中", key="shop_query")

        disabled_nearby = bool((shop_query or "").strip())

        food_types_cn = st.multiselect(
            "想吃什麼？（可多選）",
            options=list(FOOD_KEYWORDS.keys()),
            default=["餐廳（不限）"],
            disabled=disabled_nearby,
            key="food_types",
        )

        c1, c2 = st.columns(2)
        with c1:
            min_budget = st.number_input("最低預算（台幣）", min_value=0, value=0, step=50, disabled=disabled_nearby, key="min_budget")
        with c2:
            max_budget = st.number_input("最高預算（台幣）", min_value=0, value=500, step=50, disabled=disabled_nearby, key="max_budget")

        min_rating = st.slider("最低評分", 0.0, 5.0, 4.0, 0.1, disabled=disabled_nearby, key="min_rating")
        open_now = st.checkbox("只看目前營業中", value=False, disabled=disabled_nearby, key="open_now")

        submit = st.form_submit_button("🔍 開始搜尋", type="primary")

    st.markdown("---")
    st.markdown("## 🕘 搜尋歷史")
    if st.session_state.history:
        for i, h in enumerate(st.session_state.history[:10]):
            if st.button(f"⟲ {h.get('title','（未命名）')}", key=f"hist_{i}"):
                p = h.get("payload", {})
                do_search(
                    p.get("location_text", "勤美術館"),
                    p.get("shop_query", ""),
                    p.get("food_types", ["餐廳（不限）"]),
                    int(p.get("min_budget", 0)),
                    int(p.get("max_budget", 500)),
                    float(p.get("min_rating", 4.0)),
                    bool(p.get("open_now", False)),
                )
                st.rerun()
        if st.button("🧹 清空歷史", key="clear_hist"):
            st.session_state.history = []
            save_history([])
            st.rerun()
    else:
        st.caption("目前沒有紀錄。")

if not API_KEY:
    st.warning("⚠️ 尚未設定 Google API Key（GOOGLE_API_KEY / Streamlit secrets）。搜尋會失效。")

if submit:
    do_search(
        location_text,
        shop_query,
        food_types_cn,
        int(min_budget),
        int(max_budget),
        float(min_rating),
        bool(open_now),
    )

tab1, tab2, tab3, tab4 = st.tabs(["🍽️ 推薦", "📌 口袋名單", "🤖 AI", "🧭 行程"])

with tab1:
    st.subheader("推薦結果（清單 / 地圖｜右側櫥窗）")

    if st.session_state.flash:
        if hasattr(st, "toast"):
            st.toast(st.session_state.flash)
        else:
            st.success(st.session_state.flash)
        st.session_state.flash = ""

    if not st.session_state.search_done:
        st.info("先在左側搜尋一次。")
        st.stop()

    results = list(st.session_state.results)
    if not results:
        st.warning("沒有結果（可放寬評分/預算或換地點）")
        st.stop()

    with st.expander("🎛️ 篩選與排序", expanded=True):
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            min_r = st.slider("評分 ≥", 0.0, 5.0, 4.0, 0.1, key="flt_rating")
        with f2:
            max_d = st.slider("距離 ≤ (km)", 0.1, 15.0, 5.0, 0.1, key="flt_dist")
        with f3:
            min_cnt = st.number_input("評論數門檻", min_value=0, value=0, step=50, key="flt_cnt")
        with f4:
            pl_opts = [1, 2, 3, 4]
            pl_allow = st.multiselect(
                "價位",
                pl_opts,
                default=pl_opts,
                format_func=lambda x: f"{PRICE_WORD.get(int(x), str(x))}",
                key="flt_pl",
            )

        sort_mode = st.selectbox(
            "排序方式",
            ["依綜合分數", "依評分（高→低）", "依遠近（近→遠）", "依評論數（多→少）"],
            index=0,
            key="sort_mode",
        )

        view_mode = st.radio("檢視", ["清單", "地圖"], horizontal=True, key="view_mode_main")

    filtered = []
    for it in results:
        if float(it.get("rating", 0) or 0) < float(min_r):
            continue
        if float(it.get("distance_km", 9999) or 9999) > float(max_d):
            continue
        if int(it.get("user_ratings_total", 0) or 0) < int(min_cnt):
            continue
        pl = it.get("price_level", None)
        if pl is not None and int(pl) not in pl_allow:
            continue
        filtered.append(it)

    if sort_mode == "依遠近（近→遠）":
        filtered = sorted(filtered, key=lambda x: x.get("distance_km", 9999))
    elif sort_mode == "依評分（高→低）":
        filtered = sorted(filtered, key=lambda x: (x.get("rating", 0), x.get("user_ratings_total", 0)), reverse=True)
    elif sort_mode == "依評論數（多→少）":
        filtered = sorted(filtered, key=lambda x: x.get("user_ratings_total", 0), reverse=True)
    else:
        filtered = sorted(filtered, key=lambda x: x.get("score", 0), reverse=True)

    st.session_state.filtered = filtered

    pp_mode = st.selectbox("每頁顯示", ["10", "20", "30", "50", "100"], index=1, key="per_page_mode")
    per_page = int(pp_mode)
    st.session_state.per_page = int(per_page)

    total = len(filtered)
    total_pages = max(1, math.ceil(total / int(per_page)))
    st.session_state.page = min(max(1, int(st.session_state.page)), total_pages)

    ctop = st.columns([2, 2, 2, 2])
    ctop[0].markdown(f'<span class="badge badge-ok">口袋 {len(st.session_state.pocket_list)}</span>', unsafe_allow_html=True)
    ctop[1].markdown(f'<span class="badge">比較 {len(st.session_state.compare_ids)}</span>', unsafe_allow_html=True)
    ctop[2].markdown(f'<span class="badge badge-warn">行程 {len(st.session_state.trip_items)}</span>', unsafe_allow_html=True)
    ctop[3].caption(f"篩選後 {total} 筆｜第 {st.session_state.page}/{total_pages} 頁")

    nav = st.columns([1, 2, 1])
    if nav[0].button("⬅ 上一頁", key="prev_page_btn"):
        st.session_state.page = max(1, st.session_state.page - 1)
        st.rerun()
    nav[1].write(f"第 **{st.session_state.page} / {total_pages}** 頁")
    if nav[2].button("下一頁 ➡", key="next_page_btn"):
        st.session_state.page = min(total_pages, st.session_state.page + 1)
        st.rerun()

    start = (st.session_state.page - 1) * int(per_page)
    end = start + int(per_page)
    page_items = filtered[start:end]
    lookup = {x["place_id"]: x for x in filtered}

    main_col, drawer_col = st.columns([3.2, 1.6], gap="large")
    with drawer_col:
        render_drawer(st.session_state.selected_pid, st.session_state.origin)

    with main_col:
        if view_mode == "地圖":
            out = build_map(page_items, st.session_state.origin, st.session_state.selected_pid)
            pid = _extract_place_id_from_popup(out.get("last_object_clicked_popup"))
            if pid and pid != st.session_state.selected_pid:
                st.session_state.selected_pid = pid
                st.rerun()
        else:
            for it in page_items:
                pid = it["place_id"]
                name = it.get("name", "未命名店家")
                d = place_details(pid) or {}
                maps_url = d.get("url") or f"https://www.google.com/maps/place/?q=place_id:{pid}"

                thumb_bytes = None
                photos = (d.get("photos", []) or []) if d else []
                if photos:
                    ref = photos[0].get("photo_reference")
                    if ref:
                        thumb_bytes = photo_bytes(ref, maxwidth=1200)

                tags, reason, _raw = analyze_reviews(d.get("reviews", []) or [])

                oh = d.get("opening_hours", {}) if d else {}
                open_badge = ""
                if isinstance(oh, dict):
                    if oh.get("open_now") is True:
                        open_badge = '<span class="badge badge-ok">營業中</span>'
                    elif oh.get("open_now") is False:
                        open_badge = '<span class="badge badge-warn">未營業</span>'

                pl = it.get("price_level", None)
                pl_word = PRICE_WORD.get(int(pl), "未知") if pl is not None else "未知"

                with st.container(border=True):
                    top = st.columns([6, 2])
                    with top[0]:
                        st.markdown(f"### [{name}]({maps_url})")
                        if open_badge:
                            st.markdown(open_badge, unsafe_allow_html=True)

                        st.markdown(
                            f"""
<span class="badge badge-ok">⭐ {it.get('rating',0)}</span>
<span class="badge">🛣️ {float(it.get('distance_km',0) or 0):.2f} km</span>
<span class="badge">💬 評論 {it.get('user_ratings_total',0)}</span>
<span class="badge badge-warn">🏅 綜合 {float(it.get('score',0) or 0):.1f}</span>
<span class="badge">💰 {html_lib.escape(pl_word)}</span>
""",
                            unsafe_allow_html=True,
                        )

                        if thumb_bytes:
                            st.image(thumb_bytes, use_container_width=True)

                        st.write("**為什麼推薦：**")
                        st.write(reason)
                        if tags:
                            st.markdown(" ".join([f'<span class="badge">{html_lib.escape(t)}</span>' for t in tags]), unsafe_allow_html=True)

                        with st.expander("🗣️ 展開查看評論（節錄）", expanded=False):
                            reviews = d.get("reviews", []) or []
                            if not reviews:
                                st.caption("（沒有可用的文字評論）")
                            else:
                                for rv in reviews[:10]:
                                    txt = (rv.get("text", "") or "").strip()
                                    star = rv.get("rating", "")
                                    if txt:
                                        st.write(f"⭐ {star}：{txt[:520]}{'…' if len(txt)>520 else ''}")

                    with top[1]:
                        if st.button("👀 查看", key=f"view_{pid}", use_container_width=True):
                            st.session_state.selected_pid = pid
                            st.rerun()

                        if pocket_has(pid):
                            if st.button("🗑️ 移除收藏", key=f"del_{pid}", use_container_width=True):
                                pocket_remove(pid)
                                st.rerun()
                        else:
                            if st.button("➕ 收藏", key=f"add_{pid}", use_container_width=True):
                                pocket_add(pid, name)
                                st.rerun()

                        in_cmp = pid in st.session_state.compare_ids
                        if st.button("✅ 已在比較" if in_cmp else "➕ 加入比較", key=f"cmp_{pid}", use_container_width=True):
                            if not in_cmp:
                                st.session_state.compare_ids.append(pid)
                                save_compare_ids()
                            st.rerun()

                        in_trip = any(x.get("place_id") == pid for x in st.session_state.trip_items)
                        if st.button("✅ 已在行程" if in_trip else "🧭 加入行程", key=f"trip_{pid}", use_container_width=True):
                            if not in_trip:
                                st.session_state.trip_items.append({"place_id": pid, "name": name, "stay_min": 45})
                                save_trip_items()
                            st.rerun()

with tab2:
    st.subheader("📌 口袋名單（收藏＋分類）")

    pocket_items = list(st.session_state.pocket_list)
    if not pocket_items:
        st.info("你還沒有收藏任何店家。去「推薦」按 ➕ 收藏。")
        st.stop()

    cats = all_categories()
    c1, c2, c3 = st.columns([2, 3, 2])
    with c1:
        filter_cat = st.selectbox("以分類篩選", ["全部"] + cats, index=0, key="pocket_filter_cat")
    with c2:
        new_cat = st.text_input("新增分類（例如：讀書、約會、宵夜、聚會）", value="", key="pocket_new_cat")
    with c3:
        st.download_button(
            "⬇️ 匯出口袋名單 JSON",
            data=json.dumps(st.session_state.pocket_list, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="pocket_list.json",
            mime="application/json",
            key="pocket_export",
        )

    if filter_cat != "全部":
        pocket_items = [x for x in pocket_items if filter_cat in (x.get("categories", []) or [])]

    for idx, it in enumerate(pocket_items):
        pid = it["place_id"]
        d = place_details(pid) or {}
        name = d.get("name") or it.get("name") or pid
        url = d.get("url") or f"https://www.google.com/maps/place/?q=place_id:{pid}"
        addr = d.get("formatted_address", "")

        with st.container(border=True):
            top = st.columns([6, 2])
            with top[0]:
                st.markdown(f"### [{name}]({url})")
                if addr:
                    st.caption(f"📍 {addr}")
            with top[1]:
                if st.button("👀 在櫥窗查看", key=f"pocket_view_{pid}_{idx}", use_container_width=True):
                    st.session_state.selected_pid = pid
                    st.session_state.flash = "✅ 已切換櫥窗店家（回「推薦」右側櫥窗可看到）"
                    st.rerun()
                if st.button("🗑️ 移除收藏", key=f"pocket_del_{pid}_{idx}", use_container_width=True):
                    pocket_remove(pid)
                    st.rerun()

            existing = all_categories()
            default = it.get("categories", []) or []
            picked = st.multiselect(
                "分類（可多選）",
                options=sorted(set(existing + ([new_cat.strip()] if new_cat.strip() else []))),
                default=default,
                key=f"cat_sel_{pid}_{idx}",
            )
            if st.button("💾 保存分類", key=f"save_cat_{pid}_{idx}"):
                pocket_set_categories(pid, picked)
                st.session_state.flash = "✅ 已更新分類"
                st.rerun()

with tab3:
    st.subheader("🤖 AI（可用比較清單、並把結果做成可操作卡片）")

    src = st.radio("AI 參考來源", ["本次搜尋結果", "我的口袋名單"], horizontal=True, key="ai_src")
    st.session_state.ai_source = src

    pool: List[Dict[str, Any]] = []
    if src == "我的口袋名單":
        ids = [x["place_id"] for x in st.session_state.pocket_list][:180]
        for pid in ids:
            d = place_details(pid) or {}
            reviews = d.get("reviews", []) or []
            _tags, reason, _raw = analyze_reviews(reviews) if d else ([], "", "")
            geo = (d.get("geometry", {}) or {}).get("location") if d else None
            pool.append({
                "place_id": pid,
                "name": d.get("name", ""),
                "rating": float(d.get("rating", 0) or 0),
                "user_ratings_total": int(d.get("user_ratings_total", 0) or 0),
                "distance_km": 999.0,
                "score": float(d.get("rating", 0) or 0) * 22 + math.log1p(int(d.get("user_ratings_total", 0) or 0)) * 3.4,
                "reason": reason,
                "lat": float(geo["lat"]) if geo else None,
                "lng": float(geo["lng"]) if geo else None,
            })
    else:
        if st.session_state.search_done:
            for it in (st.session_state.filtered or st.session_state.results):
                pool.append({
                    "place_id": it["place_id"],
                    "name": it.get("name", ""),
                    "rating": float(it.get("rating", 0) or 0),
                    "user_ratings_total": int(it.get("user_ratings_total", 0) or 0),
                    "distance_km": float(it.get("distance_km", 0) or 0),
                    "score": float(it.get("score", 0) or 0),
                    "reason": "",
                    "lat": it.get("lat"),
                    "lng": it.get("lng"),
                })

    with st.expander("🧾 比較清單", expanded=True):
        lookup_cmp = {x["place_id"]: x for x in (st.session_state.filtered or st.session_state.results or [])}
        render_compare_manager(lookup_cmp, context_key="ai_cmp")

    for m in st.session_state.chat_messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    prompt = st.chat_input("輸入需求：例『只留安靜可讀書＋近』／『用比較清單給結論』")
    last_ai_answer = None

    if st.session_state.pending_ai_prompt:
        prompt2 = st.session_state.pending_ai_prompt
        st.session_state.pending_ai_prompt = ""
        st.session_state.chat_messages.append({"role": "user", "content": prompt2})
        with st.chat_message("assistant"):
            with st.spinner("AI 思考中…"):
                ans = openai_chat(prompt2, pool, st.session_state.chat_messages, st.session_state.compare_ids)
                st.write(ans)
                st.session_state.chat_messages.append({"role": "assistant", "content": ans})
                last_ai_answer = ans

    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            with st.spinner("AI 思考中…"):
                ans = openai_chat(prompt, pool, st.session_state.chat_messages, st.session_state.compare_ids)
                st.write(ans)
                st.session_state.chat_messages.append({"role": "assistant", "content": ans})
                last_ai_answer = ans

    if last_ai_answer:
        ai_pids = extract_place_ids_from_text(last_ai_answer)
        if ai_pids:
            st.markdown("### ✅ AI 推薦清單")
            for pid in ai_pids:
                d = place_details(pid) or {}
                name = d.get("name") or pid
                url = d.get("url") or f"https://www.google.com/maps/place/?q=place_id:{pid}"
                rating = d.get("rating", 0)
                cnt = d.get("user_ratings_total", 0)
                pl = d.get("price_level", None)
                pl_word = PRICE_WORD.get(int(pl), "未知") if pl is not None else "未知"

                with st.container(border=True):
                    st.markdown(f"### [{name}]({url})")
                    st.markdown(
                        f"""
<span class="badge badge-ok">⭐ {rating}</span>
<span class="badge">💬 評論 {cnt}</span>
<span class="badge">💰 {html_lib.escape(pl_word)}</span>
""",
                        unsafe_allow_html=True,
                    )

                    b1, b2, b3, b4 = st.columns(4)
                    if b1.button("👀 在櫥窗查看", key=f"ai_view_{pid}", use_container_width=True):
                        st.session_state.selected_pid = pid
                        st.session_state.flash = "✅ 已切換櫥窗店家（回『推薦』右側櫥窗可看到）"
                        st.rerun()

                    if pocket_has(pid):
                        if b2.button("🗑️ 移除收藏", key=f"ai_del_{pid}", use_container_width=True):
                            pocket_remove(pid); st.rerun()
                    else:
                        if b2.button("➕ 收藏", key=f"ai_add_{pid}", use_container_width=True):
                            pocket_add(pid, name); st.rerun()

                    in_cmp = pid in st.session_state.compare_ids
                    if b3.button("✅ 已在比較" if in_cmp else "➕ 加入比較", key=f"ai_cmp_{pid}", use_container_width=True):
                        if not in_cmp:
                            st.session_state.compare_ids.append(pid)
                            save_compare_ids()
                        st.rerun()

                    in_trip = any(x.get("place_id") == pid for x in st.session_state.trip_items)
                    if b4.button("✅ 已在行程" if in_trip else "🧭 加入行程", key=f"ai_trip_{pid}", use_container_width=True):
                        if not in_trip:
                            st.session_state.trip_items.append({"place_id": pid, "name": name, "stay_min": 45})
                            save_trip_items()
                        st.rerun()

    c1, c2 = st.columns(2)
    if c1.button("🧹 清空對話", key="ai_clear_chat", use_container_width=True):
        st.session_state.chat_messages = [{"role": "assistant", "content": "對話已清空。你可以重新開始問我。"}]
        st.rerun()
    if c2.button("📌 把比較清單加入口袋", key="ai_cmp_to_pocket", use_container_width=True):
        lookup2 = {x["place_id"]: x for x in (st.session_state.filtered or st.session_state.results or [])}
        for pid in list(st.session_state.compare_ids):
            if not pocket_has(pid):
                name = (lookup2.get(pid) or {}).get("name", "")
                pocket_add(pid, name)
        st.session_state.flash = "✅ 已把比較清單的店加入口袋（未重複）"
        st.rerun()

with tab4:
    lookup: Dict[str, Dict[str, Any]] = {}
    for it in (st.session_state.results or []):
        lookup[it["place_id"]] = it
    for it in (st.session_state.pocket_list or []):
        if it["place_id"] not in lookup:
            lookup[it["place_id"]] = {"place_id": it["place_id"], "name": it.get("name", ""), "lat": None, "lng": None}

    plan_trip(st.session_state.origin, lookup)
