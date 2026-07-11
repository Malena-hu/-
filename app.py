"""
app.py
校园智能挂失系统 - 双库 RAG Streamlit 入口

当前版本聚焦 RAG：
- 失物区：丢东西的人登记
- 招领区：捡东西的人登记
- 双向检索：登记后立即从对方区召回候选
"""
# 推荐启动：./.venv/bin/python -m streamlit run app.py

from __future__ import annotations

import base64
import csv
import html
import io
import os
import sys
import json
import hashlib
import secrets
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from workflow import CampusSystemCore


DEFAULT_LOCATIONS = ["一食堂前台", "二食堂前台", "图书馆失物招领处", "东门保卫室", "教学楼", "宿舍区", "操场"]
LOCATION_CONFIG_PATH = Path("./runtime_data/campus_locations.json")
USER_STORE_PATH = Path("./runtime_data/users.json")
SEARCH_SYNONYM_PATH = Path("./runtime_data/search_synonyms.json")
DEFAULT_SEARCH_SYNONYMS = {
    "充电线": ["充电线", "数据线", "线缆", "连接线", "type-c", "usb-c", "lightning", "充电器", "充电头", "适配器", "电子配件"],
    "数据线": ["数据线", "充电线", "线缆", "连接线", "type-c", "usb-c", "lightning", "电子配件"],
    "线缆": ["线缆", "充电线", "数据线", "连接线", "type-c", "usb-c", "lightning"],
    "充电器": ["充电器", "充电头", "适配器", "电源适配器", "充电线", "数据线", "电子配件"],
    "充电头": ["充电头", "充电器", "适配器", "电源适配器", "电子配件"],
    "耳机": ["耳机", "无线耳机", "蓝牙耳机", "有线耳机", "耳塞", "耳柄", "耳机盒", "充电盒", "耳机仓"],
    "无线耳机": ["无线耳机", "蓝牙耳机", "耳机", "耳塞", "耳柄", "入耳式", "半入耳式"],
    "蓝牙耳机": ["蓝牙耳机", "无线耳机", "耳机", "耳塞", "耳柄", "入耳式", "半入耳式"],
    "耳机盒": ["耳机盒", "充电盒", "耳机仓", "耳机", "盒体", "开合盖"],
    "手表": ["手表", "电子表", "运动表", "腕表", "表盘", "表带"],
    "水杯": ["水杯", "杯子", "保温杯", "杯盖", "杯身"],
    "钥匙": ["钥匙", "钥匙串", "钥匙圈"],
}
SINGLE_CHAR_ITEM_TERMS = {"伞", "笔", "书", "包", "卡", "杯"}
NO_LLM_SYNONYM_TERMS = {
    "白色",
    "黑色",
    "红色",
    "蓝色",
    "绿色",
    "黄色",
    "粉色",
    "金色",
    "银色",
    "灰色",
    "透明",
    "不限",
    "未知",
    "待认领",
    "寻找中",
    "不限",
    *DEFAULT_LOCATIONS,
}


def load_locations() -> list[str]:
    if LOCATION_CONFIG_PATH.exists():
        try:
            data = json.loads(LOCATION_CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(data, list):
                locations = [str(item).strip() for item in data if str(item).strip() and str(item).strip() != "不限"]
                return list(dict.fromkeys(locations))
        except (json.JSONDecodeError, OSError):
            pass
    return DEFAULT_LOCATIONS.copy()


def save_locations(locations: list[str]) -> None:
    cleaned = [str(item).strip() for item in locations if str(item).strip() and str(item).strip() != "不限"]
    cleaned = list(dict.fromkeys(cleaned))
    LOCATION_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCATION_CONFIG_PATH.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")


def location_options(include_any: bool = True) -> list[str]:
    locations = load_locations()
    if not locations and not include_any:
        return DEFAULT_LOCATIONS.copy()
    return ["不限", *locations] if include_any else locations


@st.cache_resource
def get_core_system() -> CampusSystemCore:
    return CampusSystemCore()


def save_uploaded_file(uploaded_file, folder: str = "./runtime_data/temp_uploads") -> str:
    save_dir = Path(folder)
    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / uploaded_file.name
    file_path.write_bytes(uploaded_file.getbuffer())
    return str(file_path)


def save_uploaded_file_unique(uploaded_file, folder: str = "./runtime_data/temp_uploads") -> str:
    save_dir = Path(folder)
    save_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded_file.name).suffix or ".jpg"
    safe_stem = Path(uploaded_file.name).stem[:40] or "upload"
    file_path = save_dir / f"{safe_stem}-{uuid.uuid4().hex[:8]}{suffix}"
    file_path.write_bytes(uploaded_file.getbuffer())
    return str(file_path)


def runtime_dependency_status() -> dict:
    status = {"python": sys.executable}
    for package in ["openai", "dashscope", "transformers", "torch"]:
        try:
            module = __import__(package)
            status[package] = getattr(module, "__version__", "ok")
        except Exception as exc:
            status[package] = f"missing: {exc}"
    return status


def load_users() -> dict:
    USER_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not USER_STORE_PATH.exists():
        return {}
    try:
        return json.loads(USER_STORE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_users(users: dict) -> None:
    USER_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    USER_STORE_PATH.write_text(json.dumps(users, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_search_term(term: str) -> str:
    return term.strip().lower()


def normalize_synonyms(values: list[str], seed: str = "") -> list[str]:
    cleaned = []
    for value in [seed, *values]:
        text = normalize_search_term(str(value))
        if not text or len(text) > 24:
            continue
        if text not in cleaned:
            cleaned.append(text)
    return cleaned[:16]


def load_search_synonyms() -> dict[str, list[str]]:
    pairs = [(key, values) for key, values in DEFAULT_SEARCH_SYNONYMS.items()]
    if SEARCH_SYNONYM_PATH.exists():
        try:
            saved = json.loads(SEARCH_SYNONYM_PATH.read_text(encoding="utf-8"))
            for key, values in saved.items():
                if isinstance(values, list):
                    pairs.append((key, values))
        except json.JSONDecodeError:
            pass
    return merge_synonym_groups(pairs)


def merge_synonym_groups(pairs: list[tuple[str, list[str]]]) -> dict[str, list[str]]:
    groups: list[dict] = []
    for key, values in pairs:
        normalized_key = normalize_search_term(key)
        terms = set(normalize_synonyms(values, normalized_key))
        if not normalized_key or not terms:
            continue
        matched_indexes = [
            index
            for index, group in enumerate(groups)
            if terms & group["terms"]
        ]
        if matched_indexes:
            first = matched_indexes[0]
            groups[first]["terms"].update(terms)
            for index in reversed(matched_indexes[1:]):
                groups[first]["terms"].update(groups[index]["terms"])
                del groups[index]
        else:
            groups.append({"key": normalized_key, "terms": set(terms)})
    return {group["key"]: sorted(group["terms"], key=lambda item: (item != group["key"], len(item), item))[:24] for group in groups}


def save_search_synonyms(synonyms: dict[str, list[str]]) -> None:
    SEARCH_SYNONYM_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean = merge_synonym_groups([(key, values) for key, values in synonyms.items()])
    SEARCH_SYNONYM_PATH.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")


def search_term_is_known(term: str, synonyms: dict[str, list[str]]) -> bool:
    normalized = normalize_search_term(term)
    if not normalized:
        return True
    for key, values in synonyms.items():
        if normalized == normalize_search_term(key) or normalized in {normalize_search_term(value) for value in values}:
            return True
    return False


def should_generate_synonyms(term: str) -> bool:
    normalized = normalize_search_term(term)
    if not normalized or normalized in NO_LLM_SYNONYM_TERMS:
        return False
    if len(normalized) < 2 and normalized not in SINGLE_CHAR_ITEM_TERMS:
        return False
    if normalized.isascii() and len(normalized) <= 5:
        return False
    return True


def generate_synonyms_with_deepseek(term: str) -> list[str]:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return []
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            timeout=20,
            max_retries=1,
        )
        system_prompt = (
            "你是校园失物招领系统的搜索同义词维护助手。"
            "你的任务是为用户搜索词生成可用于物品库筛选的中文同义词、近义词、常见别名、缩写、错写、口语说法和上位/下位物品名。"
            "你要覆盖校园常见失物场景，例如雨伞也可能被搜成伞、折叠伞、长柄伞、遮阳伞；"
            "充电线也可能被搜成数据线、type-c线、usb-c线、苹果线、充电器、充电头。"
            "只生成与校园失物物品相关的词，不要生成解释，不要生成句子，不要生成地点、颜色、状态词。"
            "输出严格 JSON，格式为 {\"term\":\"原词\", \"synonyms\":[\"词1\",\"词2\"]}，synonyms 中必须包含原词。"
        )
        user_prompt = (
            f"搜索词：{term}\n"
            "请输出 8-16 个短词。词之间应能帮助用户查到同类物品，但不要加入颜色、地点、状态。"
        )
        response = client.chat.completions.create(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.1,
            max_tokens=256,
        )
        raw = response.choices[0].message.content or ""
        start = raw.find("{")
        end = raw.rfind("}")
        data = json.loads(raw[start : end + 1] if start >= 0 and end >= start else raw)
        synonyms = data.get("synonyms", [])
        if not isinstance(synonyms, list):
            return []
        return normalize_synonyms([str(item) for item in synonyms], term)
    except Exception as exc:
        print(f"[SearchSynonyms] DeepSeek 同义词生成失败：{exc}", flush=True)
        return []


def ensure_search_synonyms(keyword: str) -> tuple[dict[str, list[str]], list[str]]:
    synonyms = load_search_synonyms()
    generated_terms = []
    keyword_parts = [part.strip() for part in keyword.replace("，", " ").replace(",", " ").split() if part.strip()]
    changed = False
    for part in keyword_parts:
        normalized = normalize_search_term(part)
        if search_term_is_known(normalized, synonyms) or not should_generate_synonyms(normalized):
            continue
        generated = generate_synonyms_with_deepseek(normalized)
        if generated:
            synonyms = merge_synonym_groups([*synonyms.items(), (normalized, generated)])
            generated_terms.append(normalized)
            changed = True
    if changed:
        save_search_synonyms(synonyms)
    return synonyms, generated_terms


def password_hash(password: str, salt: str) -> str:
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 120000)
    return digest.hex()


def register_user(role: str, account_id: str, phone: str, password: str) -> tuple[bool, str]:
    account_id = account_id.strip()
    phone = phone.strip()
    if not account_id or not phone or not password:
        return False, "请填写学号/工号、电话号码和密码。"
    users = load_users()
    key = f"{role}:{account_id}"
    if key in users:
        return False, "该身份下的账号已注册，请直接登录。"
    salt = secrets.token_hex(16)
    users[key] = {
        "role": role,
        "account_id": account_id,
        "phone": phone,
        "salt": salt,
        "password_hash": password_hash(password, salt),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_users(users)
    return True, "注册成功，请使用密码登录。"


def authenticate_user(role: str, account_id: str, password: str) -> tuple[bool, dict | None, str]:
    account_id = account_id.strip()
    users = load_users()
    user = users.get(f"{role}:{account_id}")
    if not user:
        return False, None, "账号不存在，请先注册。"
    if password_hash(password, user["salt"]) != user["password_hash"]:
        return False, None, "密码错误。"
    return True, user, "登录成功。"


def inject_css() -> None:
    st.markdown(
        """
<style>
    .stApp {
        background: #f3f6fb;
        color: #111827;
    }
    .hero {
        padding: 1.25rem 1.5rem;
        border-radius: 8px;
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 58%, #2563eb 100%);
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 14px 34px rgba(15, 23, 42, 0.18);
    }
    .hero h1 {
        margin: 0 0 .35rem 0;
        color: white;
        letter-spacing: 0;
    }
    .hero p {
        margin: 0;
        color: #dbeafe;
    }
    .flow-card {
        padding: 1rem;
        border: 1px solid #d0d5dd;
        border-radius: 8px;
        background: #ffffff;
        box-shadow: 0 1px 3px rgba(16, 24, 40, 0.08);
        margin-bottom: .85rem;
    }
    .candidate-card {
        padding: .85rem;
        border: 1px solid #e4e7ec;
        border-radius: 8px;
        background: #ffffff;
        margin-bottom: .7rem;
    }
    .pill {
        display: inline-block;
        padding: .18rem .55rem;
        border-radius: 999px;
        background: #eff4ff;
        color: #1e3a8a;
        font-size: .82rem;
        font-weight: 700;
        margin-right: .35rem;
    }
    .muted {
        color: #667085;
        font-size: .9rem;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #d0d5dd;
        border-radius: 8px;
        padding: .8rem;
        box-shadow: 0 1px 3px rgba(16, 24, 40, 0.06);
    }
    .stButton > button {
        border-radius: 6px;
        font-weight: 700;
    }
    .block-container {
        padding-top: 1.15rem;
    }
    .action-card {
        min-height: 122px;
        padding: 1rem;
        border: 1px solid #d0d5dd;
        border-radius: 8px;
        background: #ffffff;
        box-shadow: 0 1px 3px rgba(16, 24, 40, 0.06);
    }
    .action-card h3 {
        margin: 0 0 .35rem 0;
        font-size: 1.05rem;
    }
    .action-card .big {
        font-size: 2rem;
        line-height: 1.1;
        font-weight: 800;
        color: #111827;
    }
    div[data-testid="stVerticalBlockBorderWrapper"],
    div[data-testid="stVerticalBlockBorderWrapper"] *,
    div[style*="overflow: auto"],
    div[style*="overflow-y: auto"],
    div[style*="overflow: scroll"],
    div[style*="overflow-y: scroll"] {
        overscroll-behavior: contain;
    }
    .notification-drawer-title {
        padding: .6rem 0 .25rem 0;
    }
    .notification-drawer-title h2 {
        margin: 0;
        color: #111827;
    }
    .stable-record-image {
        width: 100%;
        max-height: 360px;
        object-fit: contain;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        background: #f8fafc;
    }
    .stable-record-caption {
        color: #8a94a6;
        text-align: center;
        word-break: break-all;
        margin-top: .45rem;
        font-size: .84rem;
    }
    .notification-select {
        padding-top: .2rem;
    }
    .notification-select div[data-testid="stCheckbox"] {
        min-height: 2rem;
    }
    .notification-select div[data-testid="stCheckbox"] label {
        gap: 0;
    }
    .notification-select div[data-testid="stCheckbox"] p {
        display: none;
    }
    .notification-card-warning {
        color: #991b1b;
        background: #fff1f2;
        border: 1px solid #fecdd3;
        border-radius: 999px;
        display: inline-block;
        padding: .18rem .55rem;
        font-size: .82rem;
        font-weight: 700;
        margin-bottom: .55rem;
    }
    .notification-bell-wrap {
        display: flex;
        align-items: center;
        justify-content: flex-end;
        gap: .55rem;
        margin-bottom: .25rem;
    }
    .notification-bell-icon {
        position: relative;
        width: 38px;
        height: 38px;
        border-radius: 999px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: #ffffff;
        border: 1px solid #d0d5dd;
        box-shadow: 0 2px 8px rgba(16, 24, 40, 0.08);
        color: #1f2937;
        font-size: 1.18rem;
    }
    .notification-bell-badge {
        position: absolute;
        top: -6px;
        right: -6px;
        min-width: 20px;
        height: 20px;
        padding: 0 5px;
        border-radius: 999px;
        background: #ef4444;
        color: white;
        font-size: 11px;
        line-height: 20px;
        text-align: center;
        font-weight: 800;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.claim-pinned-marker):not(:has(div[data-testid="stVerticalBlockBorderWrapper"] .claim-pinned-marker)) {
        border: 2px solid #ef4444 !important;
        background: #fffafa !important;
        box-shadow: none !important;
    }
    .scroll-center-card-marker {
        display: none;
    }
</style>
        """,
        unsafe_allow_html=True,
    )


def inject_scroll_center_script() -> None:
    components.html(
        """
<script>
(function () {
  const parentWindow = window.parent || window;
  const doc = parentWindow.document;
  if (!doc) return;

  function markCards() {
    doc.querySelectorAll('[data-scroll-center-card="1"]').forEach((marker) => {
      const wrapper = marker.closest('[data-testid="stVerticalBlockBorderWrapper"]') || marker.parentElement;
      if (wrapper) {
        wrapper.setAttribute('data-scroll-center-card-root', '1');
        wrapper.style.cursor = 'pointer';
      }
    });
  }

  function findScrollable(el) {
    let node = el.parentElement;
    let best = null;
    while (node && node !== doc.body) {
      const canScroll = node.scrollHeight > node.clientHeight + 8;
      if (canScroll) {
        best = node;
        const rect = node.getBoundingClientRect();
        if (rect.height > 180 && rect.height < parentWindow.innerHeight + 40) {
          return node;
        }
      }
      node = node.parentElement;
    }
    if (best) return best;
    return doc.scrollingElement || doc.documentElement;
  }

  function findCardFromClick(target, scroller) {
    let node = target;
    let candidate = null;
    while (node && node !== scroller && node !== doc.body) {
      if (node.nodeType !== 1) {
        node = node.parentElement;
        continue;
      }
      const rect = node.getBoundingClientRect();
      const style = parentWindow.getComputedStyle(node);
      const hasBorder = parseFloat(style.borderTopWidth || '0') > 0 || style.boxShadow !== 'none';
      const usefulSize = rect.height >= 70 && rect.width >= Math.min(260, scroller.getBoundingClientRect().width * 0.35);
      if (usefulSize && hasBorder && !candidate) {
        candidate = node;
      }
      node = node.parentElement;
    }
    return candidate || target.closest('[data-scroll-center-card-root="1"]');
  }

  function centerCard(card) {
    if (!card) return;
    const scroller = findScrollable(card);
    const cardRect = card.getBoundingClientRect();
    const isPage = scroller === doc.scrollingElement || scroller === doc.documentElement || scroller === doc.body;
    const scrollerRect = isPage
      ? { top: 0, height: parentWindow.innerHeight }
      : scroller.getBoundingClientRect();
    const visualOffset = 18;
    const delta = (cardRect.top - scrollerRect.top) - (scrollerRect.height / 2 - cardRect.height / 2) - visualOffset;
    if (isPage) {
      const pageTop = parentWindow.pageYOffset || doc.documentElement.scrollTop || doc.body.scrollTop || 0;
      parentWindow.scrollTo({ top: Math.max(0, pageTop + delta), behavior: 'smooth' });
    } else {
      scroller.scrollTo({ top: Math.max(0, scroller.scrollTop + delta), behavior: 'smooth' });
    }
  }

  function centerActiveCards() {
    doc.querySelectorAll('[data-scroll-center-active="1"]').forEach((card) => centerCard(card));
  }

  function refreshAndCenterActive() {
    markCards();
    setTimeout(centerActiveCards, 80);
    setTimeout(centerActiveCards, 260);
  }

  function scheduleActiveCenter() {
    if (!doc.querySelector('[data-scroll-center-active="1"]')) return;
    if (doc.__campusScrollCenterTimerV5) {
      parentWindow.clearTimeout(doc.__campusScrollCenterTimerV5);
    }
    doc.__campusScrollCenterTimerV5 = parentWindow.setTimeout(centerActiveCards, 140);
  }

  doc.__campusScrollCenterActiveV5 = refreshAndCenterActive;
  if (doc.__campusScrollCenterBoundV5) {
    refreshAndCenterActive();
    return;
  }
  doc.__campusScrollCenterBoundV5 = true;

  refreshAndCenterActive();
  new MutationObserver(function () {
    markCards();
    scheduleActiveCenter();
  }).observe(doc.body, { childList: true, subtree: true });
  doc.addEventListener('click', function (event) {
    if (event.target.closest && event.target.closest('button, input, textarea, select, label, [role="button"], [data-testid="stCheckbox"], [data-testid="stRadio"], [data-testid="stSelectbox"]')) {
      return;
    }
    const explicitCard = event.target.closest && event.target.closest('[data-scroll-center-card-root="1"]');
    const scroller = findScrollable(event.target);
    const card = explicitCard || findCardFromClick(event.target, scroller);
    if (!card) return;
    setTimeout(() => centerCard(card), 80);
  }, true);
})();
</script>
        """,
        height=0,
    )


def render_scroll_center_marker() -> None:
    st.markdown("<div class='scroll-center-card-marker' data-scroll-center-card='1'></div>", unsafe_allow_html=True)


def render_claim_pinned_marker() -> None:
    st.markdown("<span class='claim-pinned-marker'></span>", unsafe_allow_html=True)


def metadata_line(metadata: dict) -> str:
    parts = [
        metadata.get("category", "未知物品"),
        metadata.get("color", "未知"),
        metadata.get("location") or metadata.get("lost_location") or "未知地点",
    ]
    return " · ".join(str(part) for part in parts if part)


def render_ingest_runtime_notice(features: dict) -> None:
    vlm_source = str(features.get("vlm_source", ""))
    encoder = str(features.get("embedding_encoder", ""))
    if vlm_source.startswith("qwen_vl"):
        st.success("千问视觉模型已完成结构化特征提取；详细 JSON 仅管理员可见。")
    elif vlm_source == "identity_local":
        st.info("证件类物品已走本地脱敏路径，敏感信息不会上传云端。")
    else:
        st.warning("本次没有拿到千问视觉解析结果，系统已使用本地兜底；管理员可在 JSON 中查看 vlm_error。")

    if encoder.startswith("siglip:"):
        st.success("SigLIP 图文向量已入库。")
    else:
        st.warning("当前图文向量编码器为本地兜底 local-hash-fallback，请检查 SigLIP 模型是否已安装或可下载。")


def render_candidate_claim_actions(
    core: CampusSystemCore,
    item_id: str,
    metadata: dict,
    current_user: dict,
    key_prefix: str,
    lost_item_id: str = "",
) -> None:
    if not item_id.startswith("found-"):
        return
    if metadata.get("status") == "已认领":
        st.caption("该物品已认领。")
        return
    account_id = current_user.get("account_id", "")
    phone = current_user.get("phone", "")
    if lost_item_id:
        if st.button("已取回", key=f"{key_prefix}_picked_up_{item_id}", use_container_width=True):
            try:
                core.confirm_manual_claim(item_id, account_id, phone, lost_item_id=lost_item_id)
                pinned_ids = st.session_state.get("pinned_found_claim_ids", [])
                st.session_state["pinned_found_claim_ids"] = [pinned_id for pinned_id in pinned_ids if pinned_id != item_id]
                st.session_state.pop("upload_match_cache", None)
                st.success("已确认取回，招领记录已标记为已认领；对应寻物记录已标记为已找回。")
                st.rerun()
            except AttributeError:
                st.error("系统缓存仍是旧版本，请重启 Streamlit 服务后再试。")
            except Exception as exc:
                st.error(f"确认取回失败：{exc}")


def render_candidates(
    candidates: list[dict],
    title: str,
    empty_text: str,
    core: CampusSystemCore | None = None,
    current_user: dict | None = None,
    source_item_id: str = "",
    key_prefix: str = "candidate",
    enable_found_actions: bool = False,
) -> None:
    if title:
        st.markdown(f"### {title}")
    if not candidates:
        st.info(empty_text)
        return

    for index, candidate in enumerate(candidates, start=1):
        metadata = candidate.get("metadata", {})
        score = candidate.get("score", 0)
        with st.container(border=True):
            render_scroll_center_marker()
            top_cols = st.columns([1, 3])
            with top_cols[0]:
                image_path = metadata.get("image_path")
                if image_path and os.path.exists(image_path):
                    st.image(image_path, use_column_width=True)
                else:
                    st.caption("无图片")
            with top_cols[1]:
                st.markdown(
                    f"<span class='pill'>Top {index}</span>"
                    f"<span class='pill'>向量分 {score}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"**{metadata_line(metadata)}**")
                st.markdown(
                    f"<div class='muted'>{metadata.get('appearance') or metadata.get('distinctive_marks') or '暂无描述'}</div>",
                    unsafe_allow_html=True,
                )
                if enable_found_actions and core and current_user:
                    render_candidate_claim_actions(
                        core,
                        str(candidate.get("item_id", "")),
                        metadata,
                        current_user,
                        f"{key_prefix}_{index}",
                        lost_item_id=source_item_id,
                    )


def render_agent_decision(decision: dict | None) -> None:
    if not decision:
        return
    action = decision.get("action", "")
    notification_type = decision.get("notification_type", "none")
    if action == "notify_user":
        if notification_type == "multiple":
            st.success("DeepSeek 判断：建议发送多候选通知。")
        else:
            st.success("DeepSeek 判断：建议发送单物品通知。")
    elif action == "require_manual_review":
        st.warning("DeepSeek 判断：暂不通知，建议人工复核。")
    else:
        st.info("DeepSeek 智能通知判断未启用或无需通知。")

    st.markdown(
        f"<span class='pill'>DeepSeek置信度 {decision.get('confidence_score', 0)}</span>"
        f"<span class='pill'>动作 {decision.get('action', 'unknown')}</span>"
        f"<span class='pill'>通知类型 {notification_type}</span>"
        f"<span class='pill'>候选数 {decision.get('candidate_count', len(decision.get('candidate_item_ids') or []))}</span>",
        unsafe_allow_html=True,
    )
    if decision.get("message_title") or decision.get("message_body"):
        st.markdown(f"**通知标题：** {decision.get('message_title', '')}")
        st.write(decision.get("message_body", ""))
    if decision.get("pickup_guide"):
        st.caption(f"核验建议：{decision.get('pickup_guide')}")
    if decision.get("reason"):
        st.caption(f"判断理由：{decision.get('reason')}")


def render_lost_submission_notice(result: dict | None) -> None:
    if not result:
        return
    matches = result.get("matches") or []
    decision = result.get("agent_decision") or {}
    sent_count = len(decision.get("sent_notifications") or [])
    action = decision.get("action", "")
    notification_type = decision.get("notification_type", "none")
    candidate_count = decision.get("candidate_count", len(decision.get("candidate_item_ids") or []))

    if not matches:
        st.info("寻物登记已提交成功。系统将持续为你匹配相关招领信息，如后续发现疑似物品，会通过右上角通知提醒你及时查看。")
        return

    if action == "notify_user" or sent_count:
        if notification_type == "multiple" or candidate_count > 1:
            st.success("寻物登记已完成。系统发现多个相似候选，已为你生成通知，请到右上角通知中心逐项核验。")
        else:
            st.success("寻物登记已完成。系统发现一个高相似候选，已为你生成通知，请到右上角通知中心查看。")
        st.write("请到对应地点或联系管理人员现场确认；如果确认取回，请在下方或通知中点击“已取回”。感谢你的反馈，这会帮助系统完成闭环记录。")
        return

    if action == "require_manual_review":
        st.warning("寻物登记已完成。系统检索到了一些候选，但 DeepSeek 判断还不足以直接通知，建议后续由管理员复核；如果之后出现更明确的匹配，会通过通知提醒你。")
        return

    st.info("寻物登记已提交成功。系统将持续为你匹配相关招领信息，如后续发现疑似物品，会通过右上角通知提醒你及时查看。")


def render_lost_submission_reject_all(core: CampusSystemCore, result: dict | None) -> None:
    if not result or not result.get("matches"):
        return
    item_id = str(result.get("item_id", "latest"))
    if st.button("都不是我的", key=f"lost_submit_reject_all_{item_id}", use_container_width=True):
        decision = result.get("agent_decision") or {}
        handled_count = 0
        for message in decision.get("sent_notifications") or []:
            message_id = message.get("message_id")
            if not message_id:
                continue
            try:
                core.handle_notification_feedback(message_id, matched=False)
                handled_count += 1
            except Exception:
                continue
        suffix = f" 已同步处理 {handled_count} 条通知。" if handled_count else ""
        st.info(f"感谢您的反馈，系统匹配到后会重新为您发送通知。{suffix}")


def render_query_debug(result: dict | None) -> None:
    if not result:
        return
    debug = result.get("query_debug") or {}
    query_text = debug.get("query_text", "")
    if not query_text:
        return
    with st.expander("本次检索使用的结构化 Query"):
        st.caption(f"query_source: {debug.get('source', 'unknown')}")
        st.write(query_text)


def remember_submission(result: dict, flow_label: str) -> None:
    uploads = st.session_state.setdefault("my_uploads", [])
    features = result.get("features", {})
    uploads.insert(
        0,
        {
            "item_id": result.get("item_id"),
            "flow_label": flow_label,
            "image_path": features.get("image_path", ""),
            "summary": metadata_line(features),
            "matches": result.get("matches", []),
        },
    )
    st.session_state["my_uploads"] = uploads[:10]
    record_type = result.get("record_type")
    if record_type:
        st.session_state["selected_upload_matches"] = {
            "item_id": result.get("item_id"),
            "record_type": record_type,
        }


def image_data_uri(image_path: str) -> str:
    if not image_path or not os.path.exists(image_path):
        return ""
    suffix = Path(image_path).suffix.lower().lstrip(".")
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else "png"
    data = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{data}"


def render_upload_record_card(metadata: dict, item_id: str, flow_label: str, is_selected: bool) -> None:
    border = "#ef4444" if is_selected else "#d1d5db"
    background = "#fff5f5" if is_selected else "#ffffff"
    shadow = "0 0 0 3px rgba(239, 68, 68, 0.16)" if is_selected else "none"
    image_uri = image_data_uri(metadata.get("image_path", ""))
    image_html = (
        f"<img src='{image_uri}' alt='upload image' "
        "style='width:100%;aspect-ratio:1/1;object-fit:cover;border-radius:8px;border:1px solid #e5e7eb;'/>"
        if image_uri
        else "<div style='height:132px;border:1px dashed #cbd5e1;border-radius:8px;display:flex;align-items:center;justify-content:center;color:#64748b;'>图片不可用</div>"
    )
    selected_text = " · 当前选中" if is_selected else ""
    active_attr = "data-scroll-center-active='1'" if is_selected else ""
    st.markdown(
        f"""
        <div data-scroll-center-card-root="1" {active_attr} style="border:2px solid {border};background:{background};box-shadow:{shadow};border-radius:10px;padding:12px;margin-bottom:8px;cursor:pointer;">
          <div style="display:grid;grid-template-columns:104px 1fr;gap:12px;align-items:start;">
            <div>{image_html}</div>
            <div>
              <div style="font-weight:800;color:#111827;margin-bottom:6px;">{html.escape(flow_label + selected_text)}</div>
              <div style="color:#4b5563;font-size:0.92rem;line-height:1.45;margin-bottom:10px;">{html.escape(metadata_line(metadata))}</div>
              <span class="pill">状态 {html.escape(str(metadata.get('status', '未知')))}</span>
              <div style="color:#94a3b8;font-size:0.78rem;margin-top:8px;word-break:break-all;">编号 {html.escape(str(item_id))}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def select_upload_match(item_id: str, record_type: str) -> None:
    st.session_state["selected_upload_matches"] = {"item_id": item_id, "record_type": record_type}
    st.session_state.setdefault("upload_match_cache", {}).pop(f"{record_type}:{item_id}", None)


def render_my_uploads(core: CampusSystemCore, current_user: dict) -> None:
    account_id = current_user.get("account_id", "")
    uploads = core.list_user_items(account_id)
    st.markdown("### 我的上传记录")
    if not uploads:
        st.info("当前账号还没有历史上传记录。上传寻物或拾物图片后，会在这里查看对应的匹配排序。")
        return
    st.caption("左侧选择一条上传记录，右侧会显示它当前召回的匹配排序。")
    match_cache = st.session_state.setdefault("upload_match_cache", {})
    selected = st.session_state.get("selected_upload_matches")
    filter_label = st.radio(
        "记录类型",
        ["全部", "我的寻物记录", "我的拾物记录"],
        horizontal=True,
        key="upload_record_filter",
        label_visibility="collapsed",
    )
    filter_map = {"我的寻物记录": "lost", "我的拾物记录": "found"}
    selected_record_type = filter_map.get(filter_label)
    visible_uploads = [
        upload
        for upload in uploads
        if selected_record_type is None or upload.get("record_type") == selected_record_type
    ]

    if not visible_uploads:
        st.info(f"当前账号还没有{filter_label}。")
        return

    selected_upload = None
    if isinstance(selected, dict):
        selected_upload = next(
            (
                item
                for item in visible_uploads
                if item.get("item_id") == selected.get("item_id")
                and item.get("record_type") == selected.get("record_type")
            ),
            None,
        )

    panel_height = 640
    title_left, title_right = st.columns([1.05, 1.55], gap="large")
    with title_left:
        st.markdown(f"#### {filter_label if filter_label != '全部' else '上传记录'}")
        st.caption(f"共 {len(visible_uploads)} 条，区域内滚动浏览。")
    with title_right:
        st.markdown("#### 匹配结果")
        st.caption("选中左侧记录后，在这里查看当前召回排序。")

    left_col, right_col = st.columns([1.05, 1.55], gap="large")
    with left_col:
        with st.container(height=panel_height, border=True):
            for upload in visible_uploads:
                metadata = upload.get("metadata", {})
                item_id = upload.get("item_id")
                record_type = upload.get("record_type")
                flow_label = "拾物登记" if record_type == "found" else "寻物登记"
                is_selected = (
                    isinstance(selected, dict)
                    and selected.get("item_id") == item_id
                    and selected.get("record_type") == record_type
                )
                render_upload_record_card(metadata, item_id, flow_label, is_selected)
                st.button(
                    "查看匹配排序",
                    key=f"open_matches_{item_id}",
                    use_container_width=True,
                    on_click=select_upload_match,
                    args=(item_id, record_type),
                )

    with right_col:
        with st.container(height=panel_height, border=True):
            if not selected_upload:
                if isinstance(selected, dict):
                    st.info("当前筛选下没有选中的记录，请在左侧重新选择。")
                else:
                    st.info("点击左侧某条记录的“查看匹配排序”，结果会在这里显示。")
                return

            metadata = selected_upload.get("metadata", {})
            item_id = selected_upload.get("item_id")
            record_type = selected_upload.get("record_type")
            flow_label = "拾物登记" if record_type == "found" else "寻物登记"
            cache_key = f"stored-fused-v2:{account_id}:{record_type}:{item_id}"
            st.caption(f"当前记录：{flow_label} · {metadata_line(metadata)}")
            matches = match_cache.get(cache_key)
            error = None
            if matches is None:
                try:
                    with st.spinner("正在重新计算匹配排序..."):
                        matches = core.rematch_user_item(item_id, record_type, top_k=5)
                    match_cache[cache_key] = matches
                except Exception as exc:
                    error = exc
            if error:
                st.error(f"重新匹配失败：{error}")
            elif matches:
                render_candidates(
                    matches,
                    "",
                    "当前暂未召回相似记录。",
                    core=core,
                    current_user=current_user,
                    source_item_id=item_id if record_type == "lost" else "",
                    key_prefix=f"upload_match_{record_type}_{item_id}",
                    enable_found_actions=record_type == "lost",
                )
            else:
                st.info("当前暂未召回相似记录。")


def render_my_notifications(core: CampusSystemCore, current_user: dict) -> None:
    account_id = current_user.get("account_id", "")
    phone = current_user.get("phone", "")
    notifications = core.list_user_notifications(account_id, phone)
    st.markdown("### 我的通知")
    st.caption("DeepSeek 判断需要通知后，会在这里显示系统内通知。")
    if not notifications:
        st.info("当前账号暂时没有收到系统通知。请确认登记时已启用 DeepSeek 智能通知判断，且系统确实判断需要通知。")
        return

    with st.container(height=720, border=True):
        for item in notifications:
            with st.container(border=True):
                st.markdown(
                    f"<span class='pill'>{html.escape(str(item.get('notification_type', '通知')))}</span>"
                    f"<span class='pill'>候选数 {html.escape(str(item.get('candidate_count', 0)))}</span>"
                    f"<span class='pill'>{html.escape(str(item.get('status', 'system_sent')))}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"**{item.get('title', '疑似找到匹配物品')}**")
                if item.get("body"):
                    st.write(item.get("body"))
                if item.get("pickup_guide"):
                    st.caption(f"核验建议：{item.get('pickup_guide')}")
                st.caption(
                    f"时间：{item.get('created_at', '-')} · 来源记录：{item.get('source_record_id', '-')} · "
                    f"候选：{'、'.join(item.get('related_item_ids', []))}"
                )
                with st.expander("查看通知详情"):
                    st.json(item)


def render_record_summary(record: dict | None, title: str) -> None:
    render_scroll_center_marker()
    st.markdown(f"#### {title}")
    if not record:
        st.info("记录不存在或已被删除。")
        return
    metadata = record.get("metadata", {})
    image_path = metadata.get("image_path")
    if image_path and os.path.exists(image_path):
        image_uri = image_data_uri(image_path)
        st.markdown(
            f"""
            <img class="stable-record-image" src="{image_uri}" alt="record image"/>
            <div class="stable-record-caption">{html.escape(str(record.get("item_id", "")))}</div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.caption("图片不可用")
    st.markdown(f"**{metadata_line(metadata)}**")
    st.markdown(
        f"<span class='pill'>编号 {html.escape(str(record.get('item_id', '-')))}</span>"
        f"<span class='pill'>状态 {html.escape(str(metadata.get('status', '未知')))}</span>",
        unsafe_allow_html=True,
    )
    st.write(metadata.get("appearance") or metadata.get("distinctive_marks") or "暂无描述")


def delete_user_notifications_safely(core: CampusSystemCore, message_ids: list[str], account_id: str, phone: str = "") -> int:
    if not message_ids:
        return 0
    if hasattr(core, "delete_user_notifications"):
        return core.delete_user_notifications(message_ids, account_id, phone)

    account_id = str(account_id or "").strip()
    phone = str(phone or "").strip()
    logs = core._load_notification_logs()
    message_id_set = set(message_ids)
    kept = []
    deleted_count = 0
    for item in logs:
        is_target = item.get("message_id") in message_id_set
        recipient_name = str(item.get("recipient_name", "")).strip()
        recipient_contact = str(item.get("recipient_contact", "")).strip()
        if account_id:
            is_owner = recipient_name == account_id
        else:
            is_owner = bool(phone and not recipient_name and recipient_contact == phone)
        is_handled = item.get("status") in {"confirmed", "rejected"}
        if is_target and is_owner and is_handled:
            deleted_count += 1
            continue
        kept.append(item)
    if deleted_count:
        core._save_notification_logs(kept)
    return deleted_count


def clear_user_runtime_state() -> None:
    """切换账号时清理只属于当前用户页面的临时状态，避免信件/选中记录串号。"""
    exact_keys = {
        "user_view",
        "selected_notification_id",
        "selected_handled_notification_ids",
        "last_unread_notification_count",
        "upload_match_cache",
        "selected_upload_item_id",
    }
    prefixes = (
        "select_handled_notification_",
        "confirm_candidate_",
        "claim_candidate_",
        "cancel_claim_candidate_",
        "user_found_library_",
        "user_lost_library_",
    )
    for key in list(st.session_state.keys()):
        if key in exact_keys or any(str(key).startswith(prefix) for prefix in prefixes):
            st.session_state.pop(key, None)


def open_notification_message(core: CampusSystemCore, message_id: str) -> None:
    if not message_id:
        return
    core.mark_notification_read(message_id)
    st.session_state["selected_notification_id"] = message_id
    st.session_state["user_view"] = "notification_detail"


def get_next_notification_id(core: CampusSystemCore, current_user: dict, current_message_id: str = "") -> str:
    notifications = core.list_user_notifications(current_user.get("account_id", ""), current_user.get("phone", ""))
    if not notifications:
        return ""
    active = [
        item
        for item in notifications
        if item.get("message_id") != current_message_id and item.get("status") == "system_sent"
    ]
    fallback = [item for item in notifications if item.get("message_id") != current_message_id]
    next_pool = active or fallback
    return str(next_pool[0].get("message_id", "")) if next_pool else ""


def render_next_notification_button(
    core: CampusSystemCore,
    current_user: dict,
    key: str,
    use_container_width: bool = True,
) -> None:
    selected_id = st.session_state.get("selected_notification_id", "")
    next_id = get_next_notification_id(core, current_user, selected_id)
    if st.button("下一封信件", key=key, use_container_width=use_container_width, disabled=not bool(next_id)):
        open_notification_message(core, next_id)
        st.rerun()


def render_notification_inbox(core: CampusSystemCore, current_user: dict) -> None:
    account_id = current_user.get("account_id", "")
    phone = current_user.get("phone", "")
    notifications = core.list_user_notifications(account_id, phone)
    st.markdown("### 我的通知")
    if not notifications:
        st.info("当前账号暂时没有收到系统通知。")
        return

    read_filter = st.radio("已读状态", ["全部", "未读", "已读"], horizontal=True, key="notification_read_filter")
    handle_filter = st.radio("处理状态", ["全部", "未处理", "已处理"], horizontal=True, key="notification_handle_filter")

    filtered_notifications = []
    for item in notifications:
        unread = not item.get("read_at") and item.get("status") == "system_sent"
        handled = item.get("status") in {"confirmed", "rejected"}
        if read_filter == "未读" and not unread:
            continue
        if read_filter == "已读" and unread:
            continue
        if handle_filter == "未处理" and handled:
            continue
        if handle_filter == "已处理" and not handled:
            continue
        filtered_notifications.append(item)

    handled_ids = [
        str(item.get("message_id", ""))
        for item in filtered_notifications
        if item.get("status") in {"confirmed", "rejected"} and item.get("message_id")
    ]
    selected_delete_ids = st.session_state.setdefault("selected_handled_notification_ids", [])
    selected_delete_ids = [item for item in selected_delete_ids if item in handled_ids]
    st.session_state["selected_handled_notification_ids"] = selected_delete_ids

    def select_all_handled_notifications() -> None:
        st.session_state["selected_handled_notification_ids"] = handled_ids
        for message_id in handled_ids:
            st.session_state[f"select_handled_notification_{message_id}"] = True

    def clear_selected_handled_notifications() -> None:
        for message_id in st.session_state.get("selected_handled_notification_ids", []):
            st.session_state.pop(f"select_handled_notification_{message_id}", None)
        st.session_state["selected_handled_notification_ids"] = []

    delete_cols = st.columns([1, 1, 1.2, 2], gap="small")
    with delete_cols[0]:
        st.button(
            "全选已处理",
            use_container_width=True,
            disabled=not bool(handled_ids),
            key="select_all_handled_notifications",
            on_click=select_all_handled_notifications,
        )
    with delete_cols[1]:
        st.button(
            "清空选择",
            use_container_width=True,
            disabled=not bool(selected_delete_ids),
            key="clear_selected_handled_notifications",
            on_click=clear_selected_handled_notifications,
        )
    with delete_cols[2]:
        st.caption(f"已选择 {len(selected_delete_ids)} 封")
    with delete_cols[3]:
        if st.button(
            f"删除已选信件（{len(selected_delete_ids)}）",
            type="primary",
            use_container_width=True,
            disabled=not bool(selected_delete_ids),
            key="delete_selected_handled_notifications",
        ):
            deleted = delete_user_notifications_safely(core, selected_delete_ids, account_id, phone)
            for message_id in selected_delete_ids:
                st.session_state.pop(f"select_handled_notification_{message_id}", None)
            st.session_state["selected_handled_notification_ids"] = []
            st.success(f"已删除 {deleted} 封已处理信件。")
            st.rerun()

    st.caption(f"当前显示 {len(filtered_notifications)} / {len(notifications)} 封信件。")
    if not filtered_notifications:
        st.info("没有符合当前筛选条件的信件。")
        return

    with st.container(height=760, border=True):
        for item in filtered_notifications:
            unread = not item.get("read_at") and item.get("status") == "system_sent"
            handled = item.get("status") in {"confirmed", "rejected"}
            handled_text = "已处理" if handled else "未处理"
            message_id = item.get("message_id", "")
            with st.container(border=True):
                render_scroll_center_marker()
                if not handled:
                    st.markdown("<span class='notification-card-warning'>未处理信件不能删除，请先处理。</span>", unsafe_allow_html=True)
                selected = message_id in st.session_state["selected_handled_notification_ids"]
                card_cols = st.columns([0.08, 1], gap="small")
                with card_cols[0]:
                    if handled:
                        st.markdown("<div class='notification-select'>", unsafe_allow_html=True)
                        checkbox_key = f"select_handled_notification_{message_id}"
                        if checkbox_key not in st.session_state:
                            st.session_state[checkbox_key] = selected
                        checked = st.checkbox(
                            "选择",
                            key=checkbox_key,
                            help="选择后可批量删除",
                            label_visibility="collapsed",
                        )
                        st.markdown("</div>", unsafe_allow_html=True)
                        if checked != selected:
                            selected_ids = st.session_state.setdefault("selected_handled_notification_ids", [])
                            if checked:
                                selected_ids = [*selected_ids, message_id]
                            else:
                                selected_ids = [item_id for item_id in selected_ids if item_id != message_id]
                            st.session_state["selected_handled_notification_ids"] = selected_ids
                    else:
                        st.markdown("<div style='font-size:1.4rem;color:#cbd5e1;line-height:2.2rem;'>○</div>", unsafe_allow_html=True)
                with card_cols[1]:
                    st.markdown(
                        f"<span class='pill'>{'未读' if unread else '已读'}</span>"
                        f"<span class='pill'>{handled_text}</span>"
                        f"<span class='pill'>候选数 {html.escape(str(item.get('candidate_count', 0)))}</span>"
                        f"<span class='pill'>状态 {html.escape(str(item.get('status', '-')))}</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"**{item.get('title', '通知')}**")
                    st.caption(f"时间：{item.get('created_at', '')} · 来源记录：{item.get('source_record_id', '')}")
                    if st.button("打开信件", key=f"open_notification_{item.get('message_id')}", use_container_width=True):
                        open_notification_message(core, item.get("message_id", ""))
                        st.rerun()


def render_notification_nav(core: CampusSystemCore, current_user: dict, key_prefix: str) -> None:
    nav_cols = st.columns([1, 1, 1, 3], gap="small")
    with nav_cols[0]:
        if st.button("返回主页面", key=f"{key_prefix}_back_home", use_container_width=True):
            st.session_state["user_view"] = "lost_upload"
            st.session_state["selected_notification_id"] = ""
            st.rerun()
    with nav_cols[1]:
        if st.button("返回通知列表", key=f"{key_prefix}_back_list", use_container_width=True):
            st.session_state["user_view"] = "notifications"
            st.session_state["selected_notification_id"] = ""
            st.rerun()
    with nav_cols[2]:
        render_next_notification_button(core, current_user, f"{key_prefix}_next_notification")


def render_notification_detail_page(core: CampusSystemCore, current_user: dict) -> None:
    selected_id = st.session_state.get("selected_notification_id")
    st.markdown("### 通知详情")
    if not selected_id:
        st.info("请先从通知列表打开一封信。")
        return
    visible_message_ids = {
        item.get("message_id")
        for item in core.list_user_notifications(current_user.get("account_id", ""), current_user.get("phone", ""))
    }
    if selected_id not in visible_message_ids:
        st.session_state["selected_notification_id"] = ""
        st.warning("这封信不属于当前账号，已为你返回通知列表。")
        st.session_state["user_view"] = "notifications"
        return
    detail = core.get_notification_detail(selected_id)
    if not detail:
        st.warning("这条通知不存在或已被清理。")
        return
    message = detail["message"]
    st.markdown(
        f"<span class='pill'>{html.escape(str(message.get('notification_type', '通知')))}</span>"
        f"<span class='pill'>候选数 {html.escape(str(message.get('candidate_count', 0)))}</span>"
        f"<span class='pill'>状态 {html.escape(str(message.get('status', 'system_sent')))}</span>",
        unsafe_allow_html=True,
    )
    st.markdown(f"**{message.get('title', '疑似找到匹配物品')}**")
    st.write(message.get("body", ""))
    if message.get("pickup_guide"):
        st.caption(f"核验建议：{message.get('pickup_guide')}")

    compare_cols = st.columns([1, 1.4], gap="large")
    with compare_cols[0]:
        with st.container(height=760, border=True):
            for record in detail.get("user_records", []):
                render_record_summary(record, "我上传的物品")
    with compare_cols[1]:
        pushed_records = detail.get("pushed_records", [])
        with st.container(height=760, border=True):
            for rank, record in enumerate(pushed_records, start=1):
                render_record_summary(record, f"推送候选 Top {rank}")

    if message.get("notification_type") == "manual_claim" or message.get("direction") == "manual_claim":
        found_item_id = message.get("source_record_id", "")
        if message.get("status") in {"confirmed", "rejected"}:
            st.info("这条待办已经处理完成。")
            render_notification_nav(core, current_user, "manual_claim_handled_bottom")
            return
        action_cols = st.columns(2)
        with action_cols[0]:
            if st.button("已取回", type="primary", use_container_width=True, key=f"manual_claim_pickup_{selected_id}"):
                core.confirm_manual_claim(
                    found_item_id,
                    current_user.get("account_id", ""),
                    current_user.get("phone", ""),
                    message_id=selected_id,
                )
                st.session_state.pop("upload_match_cache", None)
                st.success("已确认取回，该招领记录已标记为已认领。")
                st.rerun()
        with action_cols[1]:
            if st.button("暂时不是我的", use_container_width=True, key=f"manual_claim_reject_{selected_id}"):
                core.handle_notification_feedback(selected_id, matched=False)
                st.warning("已从待办中处理为不是我的，系统会继续保留该招领记录。")
                st.rerun()
        render_notification_nav(core, current_user, "manual_claim_active_bottom")
        return

    if message.get("status") in {"confirmed", "rejected"}:
        st.info("这条通知已经处理完成。")
        render_notification_nav(core, current_user, "handled_detail_bottom")
        return

    pushed_records = detail.get("pushed_records", [])
    if pushed_records:
        options = {
            f"Top {index + 1} | {record.get('item_id')} | {metadata_line(record.get('metadata', {}))}": record.get("item_id")
            for index, record in enumerate(pushed_records)
        }
        selected_label = st.radio("选择已取回的推送物品", list(options.keys()), key=f"confirm_candidate_{selected_id}")
        action_cols = st.columns(2)
        with action_cols[0]:
            if st.button("已取回", type="primary", use_container_width=True, key=f"confirm_mine_{selected_id}"):
                core.handle_notification_feedback(selected_id, selected_item_id=options[selected_label], matched=True)
                st.success("已确认匹配，相关记录状态已更新。")
                st.rerun()
        with action_cols[1]:
            if st.button("都不是我的", use_container_width=True, key=f"reject_all_{selected_id}"):
                core.handle_notification_feedback(selected_id, matched=False)
                st.warning("已记录为 badcase，系统会继续等待后续匹配。")
                st.rerun()
        render_notification_nav(core, current_user, "active_detail_bottom")


def render_notification_bell(core: CampusSystemCore, current_user: dict) -> None:
    notifications = core.list_user_notifications(current_user.get("account_id", ""), current_user.get("phone", ""))
    unread_count = sum(1 for item in notifications if not item.get("read_at") and item.get("status") == "system_sent")
    previous = st.session_state.get("last_unread_notification_count", 0)
    if unread_count > previous:
        components.html(
            """
            <script>
            try {
              const ctx = new (window.AudioContext || window.webkitAudioContext)();
              const osc = ctx.createOscillator();
              const gain = ctx.createGain();
              osc.type = "sine";
              osc.frequency.value = 880;
              gain.gain.value = 0.05;
              osc.connect(gain);
              gain.connect(ctx.destination);
              osc.start();
              setTimeout(() => { osc.stop(); ctx.close(); }, 180);
            } catch (e) {}
            </script>
            """,
            height=0,
        )
    st.session_state["last_unread_notification_count"] = unread_count
    badge = f"<span class='notification-bell-badge'>{unread_count}</span>" if unread_count else ""
    bell_cols = st.columns([0.24, 1], gap="small")
    with bell_cols[0]:
        st.markdown(
            f"<div class='notification-bell-icon' title='通知'>🔔{badge}</div>",
            unsafe_allow_html=True,
        )
    with bell_cols[1]:
        if st.button("打开通知", use_container_width=True, key="open_notification_inbox"):
            st.session_state["user_view"] = "notifications"
            st.rerun()


def render_back_to_home_button() -> None:
    if st.button("返回主页面", key="notification_back_home"):
        st.session_state["user_view"] = "lost_upload"
        st.session_state["selected_notification_id"] = ""
        st.rerun()


def render_notification_toolbar(core: CampusSystemCore, current_user: dict) -> None:
    toolbar_cols = st.columns([1, 1, 1, 3], gap="small")
    with toolbar_cols[0]:
        render_back_to_home_button()
    with toolbar_cols[1]:
        if st.session_state.get("user_view") == "notification_detail":
            if st.button("返回通知列表", key="back_to_notifications"):
                st.session_state["user_view"] = "notifications"
                st.session_state["selected_notification_id"] = ""
                st.rerun()
    with toolbar_cols[2]:
        if st.session_state.get("user_view") == "notification_detail":
            render_next_notification_button(core, current_user, "top_next_notification")


def render_notification_drawer_page(core: CampusSystemCore, current_user: dict) -> None:
    side_left, workbench_col, side_right = st.columns([1, 4, 1], gap="large")
    with workbench_col:
        render_notification_toolbar(core, current_user)
        if st.session_state.get("user_view") == "notification_detail":
            render_notification_detail_page(core, current_user)
        else:
            render_notification_inbox(core, current_user)


def render_batch_rows(rows: list[dict]) -> None:
    if not rows:
        return
    success_count = sum(1 for row in rows if row["状态"] == "成功")
    if success_count == len(rows):
        st.success(f"批量处理完成：成功 {success_count} 条。")
    else:
        st.warning(f"批量处理完成：成功 {success_count} 条，失败 {len(rows) - success_count} 条。")
    st.dataframe(rows, use_container_width=True)


def library_search_text(record: dict) -> str:
    metadata = record.get("metadata", {})
    values = [record.get("item_id", ""), metadata_line(metadata)]
    for key in [
        "record_type",
        "status",
        "category",
        "subcategory",
        "main_object",
        "color",
        "material",
        "brand",
        "logo",
        "text_visible",
        "distinctive_marks",
        "shape",
        "shape_profile",
        "object_parts",
        "interface_type",
        "pair_status",
        "accessories",
        "search_keywords",
        "fine_grained_signature",
        "appearance",
        "location",
        "lost_location",
    ]:
        value = metadata.get(key, "")
        if isinstance(value, list):
            values.extend(str(item) for item in value)
        else:
            values.append(str(value))
    return " ".join(values).lower()


def expand_search_part(part: str, synonyms: dict[str, list[str]] | None = None) -> list[str]:
    part = part.strip().lower()
    if not part:
        return []
    expanded = {part}
    synonym_map = synonyms or load_search_synonyms()
    for keyword, values in synonym_map.items():
        keyword_lower = keyword.lower()
        synonym_set = {item.lower() for item in values}
        if part == keyword_lower or part in synonym_set:
            expanded.update(synonym_set)
    return sorted(expanded, key=len, reverse=True)


def library_keyword_score(record: dict, keyword_parts: list[str], synonyms: dict[str, list[str]] | None = None) -> int:
    if not keyword_parts:
        return 0
    text = library_search_text(record)
    score = 0
    for part in keyword_parts:
        expanded_parts = expand_search_part(part, synonyms)
        if part in text:
            score += 8
        matched_synonyms = [candidate for candidate in expanded_parts if candidate in text]
        if matched_synonyms:
            score += 3 + min(len(matched_synonyms), 4)
    return score


def filter_library_records(
    records: list[dict],
    keyword: str,
    location: str = "不限",
    time_range: str = "不限",
    synonyms: dict[str, list[str]] | None = None,
) -> list[dict]:
    keyword_parts = [part.strip().lower() for part in keyword.replace("，", " ").replace(",", " ").split() if part.strip()]
    synonym_map = synonyms or load_search_synonyms()
    cutoff = library_time_cutoff(time_range)
    filtered = []
    for record in records:
        metadata = record.get("metadata", {})
        if location != "不限" and metadata.get("location") != location:
            continue
        if cutoff is not None and not record_created_after(metadata, cutoff):
            continue
        text = library_search_text(record)
        if keyword_parts and not all(
            any(candidate in text for candidate in expand_search_part(part, synonym_map)) for part in keyword_parts
        ):
            continue
        filtered.append(record)
    if keyword_parts:
        filtered.sort(key=lambda record: library_keyword_score(record, keyword_parts, synonym_map), reverse=True)
    return filtered


def library_time_cutoff(time_range: str) -> datetime | None:
    days_map = {
        "今天": 1,
        "3天内": 3,
        "7天内": 7,
        "30天内": 30,
        "90天内": 90,
    }
    days = days_map.get(time_range)
    return datetime.now() - timedelta(days=days) if days else None


def record_created_after(metadata: dict, cutoff: datetime) -> bool:
    try:
        created_at = datetime.fromisoformat(str(metadata.get("created_at", "")))
    except ValueError:
        return False
    return created_at >= cutoff


def paginate_records(records: list[dict], page_size: int, page: int) -> tuple[list[dict], int, int]:
    total_pages = max(1, (len(records) + page_size - 1) // page_size)
    current_page = min(max(1, page), total_pages)
    start = (current_page - 1) * page_size
    return records[start : start + page_size], current_page, total_pages


def render_found_claim_actions(
    core: CampusSystemCore,
    item_id: str,
    metadata: dict,
    current_user: dict,
    key_prefix: str,
) -> None:
    if metadata.get("status") == "已认领":
        st.caption("该物品已认领。")
        return

    account_id = current_user.get("account_id", "")
    phone = current_user.get("phone", "")
    pinned_ids = st.session_state.setdefault("pinned_found_claim_ids", [])
    is_pinned = item_id in pinned_ids

    def toggle_found_claim_pin() -> None:
        current_pinned_ids = st.session_state.setdefault("pinned_found_claim_ids", [])
        if item_id in current_pinned_ids:
            st.session_state["pinned_found_claim_ids"] = [
                pinned_id for pinned_id in current_pinned_ids if pinned_id != item_id
            ]
        else:
            st.session_state["pinned_found_claim_ids"] = [
                item_id,
                *[pinned_id for pinned_id in current_pinned_ids if pinned_id != item_id],
            ]
            st.session_state[f"{key_prefix}_force_page"] = 1

    action_cols = st.columns(2)
    with action_cols[0]:
        pin_label = "取消标记" if is_pinned else "标记"
        st.button(
            pin_label,
            key=f"{key_prefix}_pin_{item_id}",
            use_container_width=True,
            on_click=toggle_found_claim_pin,
        )
    with action_cols[1]:
        if st.button("已取回", key=f"{key_prefix}_picked_up_{item_id}", use_container_width=True):
            try:
                core.confirm_manual_claim(item_id, account_id, phone)
                st.session_state["pinned_found_claim_ids"] = [pinned_id for pinned_id in pinned_ids if pinned_id != item_id]
                st.session_state.pop("upload_match_cache", None)
                st.success("已确认取回，该招领记录已标记为已认领。")
                st.rerun()
            except Exception as exc:
                st.error(f"确认取回失败：{exc}")


def render_found_library(
    core: CampusSystemCore,
    key_prefix: str = "found_library",
    show_admin_json: bool = False,
    current_user: dict | None = None,
) -> None:
    render_library(
        core,
        core.list_found_items(),
        "招领区现有记录",
        "招领区目前还没有记录。",
        key_prefix,
        show_admin_json,
        enable_location_filter=True,
        claim_user=None if show_admin_json else current_user,
    )


def render_lost_library(core: CampusSystemCore, key_prefix: str = "lost_library", show_admin_json: bool = False) -> None:
    render_library(
        core,
        core.list_lost_items(),
        "失物区现有记录",
        "失物区目前还没有记录。",
        key_prefix,
        show_admin_json,
        enable_location_filter=False,
    )


def render_library(
    core: CampusSystemCore,
    records: list[dict],
    title: str,
    empty_text: str,
    key_prefix: str,
    show_admin_json: bool,
    enable_location_filter: bool = False,
    claim_user: dict | None = None,
) -> None:
    st.markdown(f"### {title}")
    if not records:
        st.info(empty_text)
        return

    filter_cols = st.columns([2.2, 1.4, 1.1, 1, 1], gap="small")
    with filter_cols[0]:
        keyword = st.text_input(
            "搜索物品",
            placeholder="支持同义词搜索，如：耳机、充电线、白色、AIKE",
            key=f"{key_prefix}_keyword",
        )
    if enable_location_filter:
        configured_locations = location_options(include_any=False)
        record_locations = sorted(
            {
                str(record.get("metadata", {}).get("location", "")).strip()
                for record in records
                if str(record.get("metadata", {}).get("location", "")).strip()
            }
        )
        location_values = ["不限", *list(dict.fromkeys([*configured_locations, *record_locations]))]
        with filter_cols[1]:
            selected_location = st.selectbox("地点索引", location_values, key=f"{key_prefix}_location")
    else:
        selected_location = "不限"
        with filter_cols[1]:
            st.empty()
    with filter_cols[2]:
        time_range = st.selectbox(
            "时间索引",
            ["不限", "今天", "3天内", "7天内", "30天内", "90天内"],
            key=f"{key_prefix}_time",
        )
    with filter_cols[3]:
        page_size = st.selectbox("每页数量", [6, 10, 20, 50], index=1, key=f"{key_prefix}_page_size")

    synonym_map, generated_terms = ensure_search_synonyms(keyword)
    if generated_terms:
        st.caption(f"已为新搜索词生成并缓存同义词：{'、'.join(generated_terms)}")

    filtered_records = filter_library_records(records, keyword, selected_location, time_range, synonym_map)
    if claim_user:
        pinned_ids = st.session_state.get("pinned_found_claim_ids", [])
        pinned_order = {item_id: index for index, item_id in enumerate(pinned_ids)}
        filtered_records.sort(
            key=lambda record: (
                0 if record.get("item_id") in pinned_order else 1,
                pinned_order.get(record.get("item_id"), 999999),
            )
        )
    total_pages = max(1, (len(filtered_records) + page_size - 1) // page_size)
    page_key = f"{key_prefix}_page"
    forced_page = st.session_state.pop(f"{key_prefix}_force_page", None)
    if forced_page:
        st.session_state[page_key] = min(max(1, int(forced_page)), total_pages)
    saved_page = int(st.session_state.get(page_key, 1))
    if saved_page > total_pages:
        st.session_state[page_key] = total_pages
        saved_page = total_pages
    elif page_key not in st.session_state:
        st.session_state[page_key] = min(saved_page, total_pages)
    with filter_cols[4]:
        page = st.number_input(
            "页码",
            min_value=1,
            max_value=total_pages,
            step=1,
            key=page_key,
        )
    page_records, current_page, total_pages = paginate_records(filtered_records, page_size, int(page))

    st.caption(
        f"当前共有 {len(records)} 条记录，筛选后 {len(filtered_records)} 条；"
        f"第 {current_page}/{total_pages} 页。列表区域内滚动浏览。"
    )
    if not filtered_records:
        st.info("没有符合当前搜索或地点筛选的记录。")
        return

    with st.container(height=720, border=True):
        for record in page_records:
            metadata = record.get("metadata", {})
            item_id = record.get("item_id")
            is_pinned_claim = bool(
                claim_user and item_id in st.session_state.get("pinned_found_claim_ids", [])
            )
            with st.container(border=True):
                render_scroll_center_marker()
                if is_pinned_claim:
                    render_claim_pinned_marker()
                cols = st.columns([1, 3])
                with cols[0]:
                    image_path = metadata.get("image_path")
                    if image_path and os.path.exists(image_path):
                        st.image(image_path, caption=item_id, use_column_width=True)
                    else:
                        st.caption("图片不可用")
                with cols[1]:
                    st.markdown(f"**{metadata_line(metadata)}**")
                    st.markdown(
                        f"<span class='pill'>编号 {item_id}</span>"
                        f"<span class='pill'>状态 {metadata.get('status', '未知')}</span>",
                        unsafe_allow_html=True,
                    )
                    if show_admin_json:
                        st.markdown(
                            f"<div class='muted'>登记人：{metadata.get('operator_name', '未填写')} · "
                            f"联系方式：{metadata.get('operator_contact', '未填写')} · "
                            f"入库时间：{metadata.get('created_at', '未知')}</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"<div class='muted'>入库时间：{metadata.get('created_at', '未知')}</div>",
                            unsafe_allow_html=True,
                        )
                    st.write(metadata.get("appearance") or metadata.get("distinctive_marks") or "暂无描述")
                    if claim_user:
                        render_found_claim_actions(core, item_id, metadata, claim_user, key_prefix)
                    if show_admin_json:
                        show_json = st.checkbox("显示 JSON 特征和索引元数据", key=f"{key_prefix}_json_{item_id}")
                        if show_json:
                            st.code(json.dumps(metadata, ensure_ascii=False, indent=2), language="json")


def render_login_page() -> None:
    st.markdown(
        """
<div class="hero">
  <h1>校园智能挂失系统</h1>
  <p>请选择身份后登录。普通用户进入挂失/拾取页面，管理人员进入库维护后台。</p>
</div>
        """,
        unsafe_allow_html=True,
    )
    login_col, register_col = st.columns([1, 1])
    role_label_map = {"普通用户": "user", "管理人员": "admin"}

    with login_col:
        st.subheader("密码登录")
        login_role_label = st.radio("登录身份", list(role_label_map.keys()), horizontal=True, key="login_role")
        login_account = st.text_input("学号/工号", key="login_account")
        login_password = st.text_input("密码", type="password", key="login_password")
        if st.button("登录", type="primary", use_container_width=True):
            ok, user, message = authenticate_user(role_label_map[login_role_label], login_account, login_password)
            if ok and user:
                old_user = st.session_state.get("auth_user", {})
                if old_user.get("account_id") != user.get("account_id"):
                    clear_user_runtime_state()
                st.session_state["auth_user"] = user
                st.success(message)
                st.rerun()
            else:
                st.error(message)

    with register_col:
        st.subheader("注册账号")
        register_role_label = st.radio("注册身份", list(role_label_map.keys()), horizontal=True, key="register_role")
        register_account = st.text_input("学号/工号", key="register_account")
        register_phone = st.text_input("电话号码", key="register_phone")
        register_password = st.text_input("设置密码", type="password", key="register_password")
        register_password_confirm = st.text_input("确认密码", type="password", key="register_password_confirm")
        if st.button("注册", use_container_width=True):
            if register_password != register_password_confirm:
                st.error("两次输入的密码不一致。")
            else:
                ok, message = register_user(
                    role_label_map[register_role_label],
                    register_account,
                    register_phone,
                    register_password,
                )
                if ok:
                    st.success(message)
                else:
                    st.error(message)


def render_synonym_admin() -> None:
    st.subheader("搜索同义词记录")
    st.caption("内置词典会直接随代码加载；DeepSeek 生成的新词会缓存到 runtime_data/search_synonyms.json。")
    synonyms = load_search_synonyms()
    saved = {}
    if SEARCH_SYNONYM_PATH.exists():
        try:
            saved = json.loads(SEARCH_SYNONYM_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            saved = {}

    rows = []
    saved_keys = {normalize_search_term(key) for key in saved.keys()}
    for term, values in sorted(synonyms.items()):
        rows.append(
            {
                "搜索词": term,
                "同义词/相关词": "、".join(values),
                "来源": "DeepSeek缓存" if normalize_search_term(term) in saved_keys else "内置词典",
                "词数": len(values),
            }
        )

    search = st.text_input("筛选同义词记录", placeholder="例如：充电线、耳机、鼠标", key="admin_synonym_search")
    if search.strip():
        lowered = search.strip().lower()
        rows = [row for row in rows if lowered in row["搜索词"].lower() or lowered in row["同义词/相关词"].lower()]

    st.caption(f"当前显示 {len(rows)} 条同义词记录。")
    st.dataframe(rows, use_container_width=True, hide_index=True)

    with st.expander("查看原始 JSON"):
        st.code(json.dumps(synonyms, ensure_ascii=False, indent=2), language="json")


def render_agent_log_admin(core: CampusSystemCore) -> None:
    st.subheader("DeepSeek 智能判断记录")
    st.caption("这里记录的是可审计裁决日志：登记摘要、候选快照、最终动作、通知类型和模型返回理由。")
    logs = core.list_agent_decision_logs()
    if not logs:
        st.info("暂时没有 DeepSeek 判断记录。只有开启 DeepSeek 智能通知判断后才会写入这里。")
        return

    filter_cols = st.columns([1, 1, 1.4], gap="small")
    with filter_cols[0]:
        action_filter = st.selectbox("动作筛选", ["全部", "notify_user", "require_manual_review", "keep_waiting"], key="agent_log_action")
    with filter_cols[1]:
        notification_filter = st.selectbox("通知类型", ["全部", "single", "multiple", "none"], key="agent_log_notification")
    with filter_cols[2]:
        keyword = st.text_input("搜索记录", placeholder="编号、类别、地点、理由", key="agent_log_keyword")

    filtered = []
    lowered = keyword.strip().lower()
    for log in logs:
        decision = log.get("decision", {})
        if action_filter != "全部" and decision.get("action") != action_filter:
            continue
        if notification_filter != "全部" and decision.get("notification_type") != notification_filter:
            continue
        searchable = json.dumps(log, ensure_ascii=False).lower()
        if lowered and lowered not in searchable:
            continue
        filtered.append(log)

    rows = []
    for log in filtered:
        decision = log.get("decision", {})
        rows.append(
            {
                "时间": log.get("created_at", ""),
                "登记编号": log.get("record_id", ""),
                "方向": log.get("direction", ""),
                "候选数": log.get("candidate_count", 0),
                "动作": decision.get("action", ""),
                "通知类型": decision.get("notification_type", ""),
                "通知候选数": decision.get("candidate_count", len(decision.get("candidate_item_ids") or [])),
                "置信度": decision.get("confidence_score", 0),
                "理由": decision.get("reason", ""),
            }
        )

    st.caption(f"当前显示 {len(rows)} 条，日志最多保留最近 300 条。")
    st.dataframe(rows, use_container_width=True, hide_index=True)

    options = {
        f"{log.get('created_at', '')} | {log.get('record_id', '')} | {log.get('decision', {}).get('notification_type', '')}": log
        for log in filtered[:80]
    }
    if options:
        selected = st.selectbox("查看单条详情", list(options.keys()), key="agent_log_detail")
        log = options[selected]
        decision = log.get("decision", {})
        st.markdown("#### 裁决摘要")
        st.write(
            {
                "动作": decision.get("action"),
                "通知类型": decision.get("notification_type"),
                "通知候选数": decision.get("candidate_count", len(decision.get("candidate_item_ids") or [])),
                "置信度": decision.get("confidence_score"),
                "候选ID": decision.get("candidate_item_ids"),
                "理由": decision.get("reason"),
            }
        )
        with st.expander("查看候选快照"):
            st.json(log.get("candidate_snapshot", []))
        with st.expander("查看完整审计 JSON"):
            st.json(log)


def render_notification_admin(core: CampusSystemCore) -> None:
    st.subheader("通知发送记录")
    st.caption("DeepSeek 最终裁决为通知后，系统会在这里记录已发送的站内通知；后续接短信/微信时可复用这些消息。")
    logs = core.list_notification_logs()
    if not logs:
        st.info("暂时没有通知发送记录。")
        return

    filter_cols = st.columns([1, 1, 1.4], gap="small")
    with filter_cols[0]:
        notification_filter = st.selectbox("通知类型", ["全部", "single", "multiple"], key="notification_log_type")
    with filter_cols[1]:
        channel_filter = st.selectbox("发送通道", ["全部", "in_app"], key="notification_log_channel")
    with filter_cols[2]:
        keyword = st.text_input("搜索通知", placeholder="联系人、编号、正文", key="notification_log_keyword")

    lowered = keyword.strip().lower()
    filtered = []
    for log in logs:
        if notification_filter != "全部" and log.get("notification_type") != notification_filter:
            continue
        if channel_filter != "全部" and log.get("channel") != channel_filter:
            continue
        searchable = json.dumps(log, ensure_ascii=False).lower()
        if lowered and lowered not in searchable:
            continue
        filtered.append(log)

    rows = [
        {
            "时间": log.get("created_at", ""),
            "联系人": log.get("recipient_name", ""),
            "联系方式": log.get("recipient_contact", ""),
            "通知类型": log.get("notification_type", ""),
            "候选数": log.get("candidate_count", 0),
            "来源记录": log.get("source_record_id", ""),
            "相关候选": "、".join(log.get("related_item_ids", [])),
            "状态": log.get("status", ""),
            "标题": log.get("title", ""),
        }
        for log in filtered
    ]
    st.caption(f"当前显示 {len(rows)} 条，最多保留最近 500 条。")
    st.dataframe(rows, use_container_width=True, hide_index=True)
    render_export_buttons("notification_logs", filtered, rows)
    if filtered:
        selected = st.selectbox(
            "查看通知详情",
            [f"{log.get('created_at', '')} | {log.get('recipient_contact', '')} | {log.get('source_record_id', '')}" for log in filtered[:80]],
            key="notification_log_detail",
        )
        selected_index = list(
            f"{log.get('created_at', '')} | {log.get('recipient_contact', '')} | {log.get('source_record_id', '')}"
            for log in filtered[:80]
        ).index(selected)
        st.json(filtered[selected_index])


def render_export_buttons(prefix: str, raw_rows: list[dict], table_rows: list[dict]) -> None:
    export_cols = st.columns(2)
    with export_cols[0]:
        st.download_button(
            "导出 JSON",
            data=json.dumps(raw_rows, ensure_ascii=False, indent=2),
            file_name=f"{prefix}.json",
            mime="application/json",
            use_container_width=True,
            key=f"{prefix}_json_download",
        )
    with export_cols[1]:
        buffer = io.StringIO()
        if table_rows:
            writer = csv.DictWriter(buffer, fieldnames=list(table_rows[0].keys()))
            writer.writeheader()
            writer.writerows(table_rows)
        st.download_button(
            "导出 CSV",
            data=buffer.getvalue(),
            file_name=f"{prefix}.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"{prefix}_csv_download",
        )


def render_resolution_admin(core: CampusSystemCore) -> None:
    st.subheader("匹配闭环与 Badcase 记录")
    st.caption("这里保留成功认领和用户否认的完整链路：向量召回分数、Agent 精排、推送通知、用户最终选择。")
    logs = core.list_resolution_logs()
    if not logs:
        st.info("暂时没有闭环记录。")
        return
    result_filter = st.selectbox("处理结果", ["全部", "matched", "badcase"], key="resolution_result_filter")
    filtered = [item for item in logs if result_filter == "全部" or item.get("result") == result_filter]
    rows = [
        {
            "时间": item.get("created_at", ""),
            "结果": item.get("result", ""),
            "通知ID": item.get("message_id", ""),
            "来源记录": item.get("source_record_id", ""),
            "用户选择候选": item.get("selected_item_id", ""),
            "候选排名": item.get("selected_rank", ""),
            "相关候选": "、".join(item.get("related_item_ids", [])),
        }
        for item in filtered
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)
    render_export_buttons("resolution_logs", filtered, rows)
    if filtered:
        selected = st.selectbox(
            "查看闭环详情",
            [f"{item.get('created_at', '')} | {item.get('result', '')} | {item.get('message_id', '')}" for item in filtered[:100]],
            key="resolution_detail",
        )
        selected_index = list(
            f"{item.get('created_at', '')} | {item.get('result', '')} | {item.get('message_id', '')}"
            for item in filtered[:100]
        ).index(selected)
        st.json(filtered[selected_index])


def render_location_admin(core: CampusSystemCore) -> None:
    st.markdown("#### 校园地点索引管理")
    st.caption("这里维护登记和筛选下拉中的校园地点；已有记录中的历史地点仍会在库筛选中保留。")
    configured_locations = load_locations()
    record_locations = sorted(
        {
            str(metadata.get("location") or metadata.get("lost_location") or "").strip()
            for record in [*core.list_found_items(), *core.list_lost_items()]
            for metadata in [record.get("metadata", {})]
            if str(metadata.get("location") or metadata.get("lost_location") or "").strip()
            and str(metadata.get("location") or metadata.get("lost_location") or "").strip() != "未知地点"
        }
    )
    st.write(
        {
            "当前配置地点数": len(configured_locations),
            "记录中出现过的地点数": len(record_locations),
        }
    )

    add_col, remove_col = st.columns(2)
    with add_col:
        new_location = st.text_input("新增地点", placeholder="例如：三食堂服务台", key="admin_new_location")
        if st.button("添加地点", use_container_width=True, key="admin_add_location"):
            location = new_location.strip()
            if not location or location == "不限":
                st.warning("请输入有效地点名称。")
            elif location in configured_locations:
                st.info("这个地点已经存在。")
            else:
                save_locations([*configured_locations, location])
                st.success(f"已添加地点：{location}")
                st.rerun()

    with remove_col:
        removable = configured_locations.copy()
        remove_location = st.selectbox(
            "删除地点配置",
            removable,
            key="admin_remove_location",
            disabled=not removable,
        )
        if st.button("删除选中地点", use_container_width=True, key="admin_delete_location", disabled=not removable):
            save_locations([item for item in configured_locations if item != remove_location])
            st.success(f"已从配置中删除：{remove_location}。已有记录不会被删除。")
            st.rerun()

    with st.expander("查看当前地点配置和历史记录地点"):
        st.json({"configured_locations": configured_locations, "record_locations": record_locations})


def render_bulk_delete_panel(
    title: str,
    records: list[dict],
    key_prefix: str,
    delete_fn,
    empty_text: str,
    allowed_statuses: list[str],
) -> None:
    st.markdown(f"**{title}**")
    if not records:
        st.info(empty_text)
        return

    statuses = sorted(
        {
            str(item.get("metadata", {}).get("status", "")).strip()
            for item in records
            if str(item.get("metadata", {}).get("status", "")).strip()
        }
    )
    status_options = ["全部", *allowed_statuses]
    status_options.extend(status for status in statuses if status not in status_options)

    filter_cols = st.columns([1.3, 1, 1], gap="small")
    with filter_cols[0]:
        keyword = st.text_input("筛选关键词", placeholder="编号、类别、颜色、地点", key=f"{key_prefix}_delete_keyword")
    with filter_cols[1]:
        age_label = st.selectbox(
            "时间范围",
            ["不限", "7天以前", "30天以前", "90天以前", "180天以前"],
            key=f"{key_prefix}_delete_age",
        )
    with filter_cols[2]:
        status_filter = st.selectbox("物品状态", status_options, key=f"{key_prefix}_delete_status")

    age_days_map = {"7天以前": 7, "30天以前": 30, "90天以前": 90, "180天以前": 180}
    cutoff = datetime.now() - timedelta(days=age_days_map[age_label]) if age_label in age_days_map else None
    keyword_parts = [
        part.strip().lower()
        for part in keyword.replace("，", " ").replace(",", " ").split()
        if part.strip()
    ]

    filtered_records = []
    for item in records:
        metadata = item.get("metadata", {})
        if status_filter != "全部" and metadata.get("status") != status_filter:
            continue
        if cutoff is not None:
            try:
                created_at = datetime.fromisoformat(str(metadata.get("created_at", "")))
                if created_at > cutoff:
                    continue
            except ValueError:
                continue
        search_text = library_search_text(item)
        if keyword_parts and not all(part in search_text for part in keyword_parts):
            continue
        filtered_records.append(item)

    select_all = st.checkbox(
        f"全选当前筛选结果（{len(filtered_records)} 条）",
        key=f"{key_prefix}_select_all",
        disabled=not filtered_records,
    )

    selected_ids = []
    st.caption(f"当前筛选出 {len(filtered_records)} 条。")
    if not filtered_records:
        st.info("没有符合当前筛选条件的记录。")
        return

    with st.container(height=360, border=True):
        for item in filtered_records:
            item_id = item["item_id"]
            metadata = item.get("metadata", {})
            label = f"{item_id} | {metadata_line(metadata)} | 状态 {metadata.get('status', '未知')} | {metadata.get('created_at', '未知时间')}"
            checked = st.checkbox(
                label,
                value=select_all,
                disabled=select_all,
                key=f"{key_prefix}_row_{item_id}",
            )
            if select_all or checked:
                selected_ids.append(item_id)

    st.caption(f"已选择 {len(selected_ids)} 条。")
    if selected_ids:
        with st.expander("查看已选记录"):
            for item in filtered_records:
                if item["item_id"] in selected_ids:
                    st.write(f"{item['item_id']} | {metadata_line(item.get('metadata', {}))}")

    if st.button(
        f"批量删除选中的 {len(selected_ids)} 条记录",
        type="primary",
        use_container_width=True,
        disabled=not selected_ids,
        key=f"{key_prefix}_delete_button",
    ):
        for item_id in selected_ids:
            delete_fn(item_id)
        st.success(f"已删除 {len(selected_ids)} 条记录。")
        st.rerun()


def render_clear_all_records_panel(core: CampusSystemCore, found_count: int, lost_count: int) -> None:
    clear_all_nonce = st.session_state.setdefault("admin_clear_all_nonce", 0)

    st.markdown("### 一键清空系统业务记录")
    st.warning(
        "危险操作：将清空失物区、招领区、DeepSeek审计记录、通知记录和闭环记录。"
        "账号、地点索引、同义词词典和 API Key 不会被删除。"
    )
    clear_result = st.session_state.pop("admin_clear_all_result", None)
    if clear_result:
        st.success(
            "系统记录已清空："
            f"招领区 {clear_result['found_records']} 条，失物区 {clear_result['lost_records']} 条；"
            f"DeepSeek记录 {clear_result['logs'].get('agent_decision_logs', 0)} 条，"
            f"通知记录 {clear_result['logs'].get('notification_messages', 0)} 条，"
            f"闭环记录 {clear_result['logs'].get('resolution_logs', 0)} 条。"
        )
        if clear_result.get("_cleanup_images"):
            if clear_result.get("failed_upload_files"):
                st.warning(
                    f"已删除上传图片 {clear_result.get('removed_upload_files', 0)} 个，"
                    f"{clear_result.get('failed_upload_files', 0)} 个文件删除失败。"
                )
            else:
                st.info(f"已删除上传图片 {clear_result.get('removed_upload_files', 0)} 个。")

    cleanup_images = st.checkbox(
        "同时删除记录关联的上传图片文件",
        value=False,
        key="admin_clear_all_cleanup_images",
        help="默认只清空系统记录。勾选后会尝试删除这些记录引用的上传图片文件。",
    )
    confirm_text = st.text_input(
        "请输入“确认清空”以启用按钮",
        key=f"admin_clear_all_confirm_text_{clear_all_nonce}",
        placeholder="确认清空",
    )
    disabled = confirm_text.strip() != "确认清空" or (found_count + lost_count == 0)
    if st.button(
        f"一键删除所有系统记录（当前 {found_count + lost_count} 条）",
        type="primary",
        use_container_width=True,
        disabled=disabled,
        key="admin_clear_all_records_button",
    ):
        result = core.clear_all_system_records(remove_upload_files=cleanup_images)
        for key in [
            "pinned_found_claim_ids",
            "upload_match_cache",
            "selected_handled_notification_ids",
            "selected_notification_id",
        ]:
            st.session_state.pop(key, None)
        result["_cleanup_images"] = cleanup_images
        st.session_state["admin_clear_all_result"] = result
        st.session_state["admin_clear_all_nonce"] = clear_all_nonce + 1
        st.rerun()


def render_vector_index_repair_panel(core: CampusSystemCore) -> None:
    st.markdown("### 修复 Chroma 向量索引")
    st.caption("以 JSON 业务记录为准重建 Chroma。适合出现“记录存在但匹配为空”、Chroma ID 与业务记录不同步时使用。")
    consistency = core.inspect_vector_index_consistency()
    rows = []
    for label, key in [("招领区", "found"), ("失物区", "lost")]:
        info = consistency.get(key, {})
        rows.append(
            {
                "库": label,
                "JSON记录数": info.get("json_count", 0),
                "Chroma记录数": info.get("chroma_count", 0),
                "JSON有但Chroma缺失": len(info.get("missing_in_chroma", [])),
                "Chroma有但JSON缺失": len(info.get("extra_in_chroma", [])),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)

    rebuild_result = st.session_state.pop("vector_index_rebuild_result", None)
    if rebuild_result:
        found = rebuild_result.get("found", {})
        lost = rebuild_result.get("lost", {})
        st.success(
            "Chroma 索引已重建："
            f"招领区写入 {found.get('rebuilt_count', 0)} 条、跳过 {found.get('skipped_count', 0)} 条；"
            f"失物区写入 {lost.get('rebuilt_count', 0)} 条、跳过 {lost.get('skipped_count', 0)} 条。"
        )
        failed = found.get("failed_records", []) + lost.get("failed_records", [])
        if failed:
            with st.expander("查看跳过/失败记录"):
                st.json(failed)

    if st.button("按 JSON 重建 Chroma 索引", use_container_width=True, key="rebuild_chroma_index_button"):
        st.session_state["vector_index_rebuild_result"] = core.rebuild_vector_indexes()
        st.rerun()


def logout_button() -> None:
    user = st.session_state.get("auth_user", {})
    role_text = "管理人员" if user.get("role") == "admin" else "普通用户"
    st.sidebar.markdown(f"**当前身份：** {role_text}")
    st.sidebar.caption(f"账号：{user.get('account_id', '-')}")
    if st.sidebar.button("退出登录", use_container_width=True):
        clear_user_runtime_state()
        st.session_state.pop("auth_user", None)
        st.rerun()


def render_admin_page(core: CampusSystemCore) -> None:
    status = core.get_system_status()
    st.markdown(
        """
<div class="hero">
  <h1>管理人员后台</h1>
  <p>查看并维护失物区和招领区。当前开放无效记录删除、库内容查看、同义词记录和系统状态检查。</p>
</div>
        """,
        unsafe_allow_html=True,
    )
    logout_button()

    metric_cols = st.columns(4)
    metric_cols[0].metric("失物区", status["lost_count"])
    metric_cols[1].metric("招领区", status["found_count"])
    metric_cols[2].metric("向量后端", status.get("vector_backend", "unknown"))
    metric_cols[3].metric("Agent 状态", status.get("agent_mode", "unknown"))

    overview_tab, found_tab, lost_tab, bulk_tab, locations_tab, synonyms_tab, agent_log_tab, notification_tab, resolution_tab, maintenance_tab = st.tabs(
        ["系统概览", "招领区", "失物区", "批量入库", "地点索引", "同义词记录", "DeepSeek记录", "通知记录", "闭环记录", "维护操作"]
    )

    with overview_tab:
        st.json(status)
        st.markdown("#### 运行环境诊断")
        st.json(runtime_dependency_status())

    with locations_tab:
        render_location_admin(core)

    with found_tab:
        render_found_library(core, key_prefix="admin_found_library", show_admin_json=True)

    with lost_tab:
        render_lost_library(core, key_prefix="admin_lost_library", show_admin_json=True)

    with bulk_tab:
        st.subheader("管理员批量上传图片入库")
        st.caption("批量入库会逐张调用千问视觉模型提取 JSON，并写入对应向量库；重复图片会被跳过并显示失败原因。")

        target_library = st.radio(
            "选择入库目标",
            ["招领区", "失物区"],
            horizontal=True,
            key="admin_bulk_target",
        )
        bulk_files = st.file_uploader(
            "批量选择图片",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="admin_bulk_files",
        )

        if target_library == "招领区":
            bulk_location = st.selectbox("统一存放地点", location_options(include_any=False), key="admin_bulk_found_location")
            bulk_note = st.text_input("统一拾物备注（可选）", placeholder="例如：管理员批量导入、来自失物招领处整理")
        else:
            bulk_location = st.selectbox("统一可能丢失地点", location_options(include_any=True), key="admin_bulk_lost_location")
            bulk_note = st.text_input("统一寻物描述补充（可选）", placeholder="例如：管理员代录历史寻物图片")

        admin_name = st.session_state.get("auth_user", {}).get("account_id", "admin")
        admin_contact = st.session_state.get("auth_user", {}).get("phone", "")
        top_k = st.slider("每张图片召回候选数量", min_value=1, max_value=10, value=5, key="admin_bulk_top_k")
        admin_use_agent = st.checkbox("批量入库时启用 DeepSeek 智能通知判断", value=True, key="admin_bulk_use_agent")

        if st.button("开始批量入库", type="primary", use_container_width=True, disabled=not bulk_files):
            rows = []
            progress = st.progress(0)
            for index, uploaded_file in enumerate(bulk_files, start=1):
                try:
                    temp_path = save_uploaded_file_unique(uploaded_file, "./runtime_data/admin_bulk_uploads")
                    if target_library == "招领区":
                        result = core.report_found_item(
                            image_path=temp_path,
                            location=bulk_location,
                            reporter_note=bulk_note.strip(),
                            operator_name=admin_name,
                            operator_contact=admin_contact,
                            top_k=top_k,
                            use_agent=admin_use_agent,
                        )
                    else:
                        location = "" if bulk_location == "不限" else bulk_location
                        result = core.report_lost_item(
                            description=bulk_note.strip() or Path(uploaded_file.name).stem,
                            lost_location=location,
                            contact=admin_contact,
                            query_image_path=temp_path,
                            operator_name=admin_name,
                            operator_contact=admin_contact,
                            top_k=top_k,
                            use_agent=admin_use_agent,
                        )
                    decision = result.get("agent_decision", {})
                    rows.append(
                        {
                            "文件": uploaded_file.name,
                            "状态": "成功",
                            "编号": result["item_id"],
                            "摘要": metadata_line(result.get("features", {})),
                            "候选数": len(result.get("matches", [])),
                            "Agent动作": decision.get("action", "未启用"),
                            "通知类型": decision.get("notification_type", "none"),
                            "失败原因": "",
                        }
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "文件": uploaded_file.name,
                            "状态": "失败",
                            "编号": "",
                            "摘要": "",
                            "候选数": 0,
                            "Agent动作": "",
                            "通知类型": "",
                            "失败原因": str(exc),
                        }
                    )
                progress.progress(index / len(bulk_files))
            st.session_state["admin_bulk_results"] = rows
            success_count = sum(1 for row in rows if row["状态"] == "成功")
            st.success(f"批量入库完成：成功 {success_count} 张，失败 {len(rows) - success_count} 张。")
            st.dataframe(rows, use_container_width=True)

        if st.session_state.get("admin_bulk_results"):
            st.markdown("#### 最近一次批量入库结果")
            st.dataframe(st.session_state["admin_bulk_results"], use_container_width=True)

    with synonyms_tab:
        render_synonym_admin()

    with agent_log_tab:
        render_agent_log_admin(core)

    with notification_tab:
        render_notification_admin(core)

    with resolution_tab:
        render_resolution_admin(core)

    with maintenance_tab:
        st.subheader("系统记录维护")
        found_records = core.list_found_items()
        lost_records = core.list_lost_items()
        st.subheader("删除无效/无用记录")

        col_found, col_lost = st.columns(2)
        with col_found:
            render_bulk_delete_panel(
                "招领区删除",
                found_records,
                "delete_found",
                core.delete_found_item,
                "招领区暂无记录。",
                ["待认领", "已认领"],
            )

        with col_lost:
            render_bulk_delete_panel(
                "失物区删除",
                lost_records,
                "delete_lost",
                core.delete_lost_item,
                "失物区暂无记录。",
                ["寻找中", "已找回"],
            )

        st.divider()
        render_vector_index_repair_panel(core)
        st.divider()
        render_clear_all_records_panel(core, len(found_records), len(lost_records))


st.set_page_config(
    page_title="校园智能挂失系统",
    page_icon="🎒",
    layout="wide",
)
inject_css()
inject_scroll_center_script()

if "auth_user" not in st.session_state:
    render_login_page()
    st.stop()

core = get_core_system()
required_core_methods = [
    "create_manual_claim_notification",
    "confirm_manual_claim",
    "clear_all_system_records",
    "inspect_vector_index_consistency",
    "rebuild_vector_indexes",
]
if any(not hasattr(core, method_name) for method_name in required_core_methods):
    get_core_system.clear()
    core = get_core_system()
status = core.get_system_status()
current_user = st.session_state["auth_user"]

if st.session_state["auth_user"].get("role") == "admin":
    render_admin_page(core)
    st.stop()

if "user_view" not in st.session_state:
    st.session_state["user_view"] = "lost_upload"

if st.session_state["user_view"] in {"notifications", "notification_detail"}:
    render_notification_drawer_page(core, current_user)
    st.stop()

bell_spacer, bell_col = st.columns([4.4, 1.6])
with bell_col:
    render_notification_bell(core, current_user)

upload_count = len(core.list_user_items(current_user.get("account_id", "")))
entry_cols = st.columns(5)
with entry_cols[0]:
    st.markdown("<div class='action-card'><h3>我丢了东西</h3><div class='big'>上传寻物</div></div>", unsafe_allow_html=True)
    if st.button("进入寻物登记", type="primary", use_container_width=True):
        st.session_state["user_view"] = "lost_upload"
with entry_cols[1]:
    st.markdown("<div class='action-card'><h3>我捡到东西</h3><div class='big'>上传拾物</div></div>", unsafe_allow_html=True)
    if st.button("进入拾物登记", type="primary", use_container_width=True):
        st.session_state["user_view"] = "found_upload"
with entry_cols[2]:
    st.markdown(
        f"<div class='action-card'><h3>失物区</h3><div class='big'>{status['lost_count']}</div></div>",
        unsafe_allow_html=True,
    )
    if st.button("查看失物区", use_container_width=True):
        st.session_state["user_view"] = "lost_library"
with entry_cols[3]:
    st.markdown(
        f"<div class='action-card'><h3>招领区</h3><div class='big'>{status['found_count']}</div></div>",
        unsafe_allow_html=True,
    )
    if st.button("查看招领区", use_container_width=True):
        st.session_state["user_view"] = "found_library"
with entry_cols[4]:
    st.markdown(
        f"<div class='action-card'><h3>我的上传记录</h3><div class='big'>{upload_count}</div></div>",
        unsafe_allow_html=True,
    )
    if st.button("查看上传记录", use_container_width=True):
        st.session_state["user_view"] = "my_uploads"

with st.sidebar:
    logout_button()

if st.session_state["user_view"] == "lost_upload":
    st.subheader("丢东西的人：登记失物信息，并检索招领区")
    st.write("填写物品描述和可能丢失地点。提交后，这条记录会进入“失物区”，同时从“招领区”召回相似候选。")

    form_cols = st.columns([2, 1])
    with form_cols[0]:
        lost_description = st.text_area(
            "物品描述",
            placeholder="例如：我在二食堂附近丢了一个蓝色水杯，杯身有熊猫贴纸",
            height=130,
        )
        contact = st.text_input(
            "联系方式或备注",
            value=current_user.get("phone", ""),
            placeholder="例如：手机号后四位 / 微信 / 班级",
            key=f"lost_contact_{current_user.get('role', 'user')}_{current_user.get('account_id', '')}",
        )
        lost_images = st.file_uploader(
            "历史图片（可选，支持多选）",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="lost_images",
        )
    with form_cols[1]:
        lost_location = st.selectbox("可能丢失地点", location_options(include_any=True), index=0, key="lost_location")
        lost_top_k = st.slider("召回候选数量", min_value=1, max_value=10, value=5, key="lost_top_k")
        lost_use_agent = st.checkbox("启用 DeepSeek 智能通知判断", value=True, key="lost_use_agent")
        submit_lost = st.button("登记寻物并检索", type="primary", use_container_width=True)

    if submit_lost:
        if not lost_description.strip() and not lost_images:
            st.warning("请至少填写物品描述或上传历史图片。")
        else:
            location = "" if lost_location == "不限" else lost_location
            uploaded_list = lost_images or [None]
            rows = []
            latest_result = None
            with st.spinner("正在写入失物区，并检索招领区..."):
                progress = st.progress(0) if len(uploaded_list) > 1 else None
                for index, uploaded_file in enumerate(uploaded_list, start=1):
                    try:
                        image_path = (
                            save_uploaded_file_unique(uploaded_file, "./runtime_data/temp_lost_uploads")
                            if uploaded_file
                            else None
                        )
                        description = lost_description.strip()
                        if uploaded_file and not description:
                            description = Path(uploaded_file.name).stem
                        result = core.report_lost_item(
                            description=description,
                            lost_location=location,
                            contact=contact.strip(),
                            query_image_path=image_path,
                            operator_name=current_user.get("account_id", ""),
                            operator_contact=contact.strip() or current_user.get("phone", ""),
                            top_k=lost_top_k,
                            use_agent=lost_use_agent,
                        )
                        latest_result = result
                        st.session_state["selected_upload_matches"] = {
                            "item_id": result["item_id"],
                            "record_type": "lost",
                        }
                        rows.append(
                            {
                                "文件": uploaded_file.name if uploaded_file else "文字描述",
                                "状态": "成功",
                                "编号": result["item_id"],
                                "候选数": len(result.get("matches", [])),
                                "失败原因": "",
                            }
                        )
                    except Exception as exc:
                        rows.append(
                            {
                                "文件": uploaded_file.name if uploaded_file else "文字描述",
                                "状态": "失败",
                                "编号": "",
                                "候选数": 0,
                                "失败原因": str(exc),
                            }
                        )
                    if progress:
                        progress.progress(index / len(uploaded_list))
            failed_rows = [row for row in rows if row.get("状态") == "失败"]
            if failed_rows:
                first_error = failed_rows[0].get("失败原因", "未知错误")
                st.error(f"有 {len(failed_rows)} 条寻物登记失败：{first_error}")
            if latest_result:
                render_lost_submission_notice(latest_result)
                render_candidates(
                    latest_result.get("matches", []),
                    "系统为你召回的招领候选",
                    "暂未在招领区中找到相似物品。",
                    core=core,
                    current_user=current_user,
                    source_item_id=latest_result.get("item_id", ""),
                    key_prefix=f"lost_submit_{latest_result.get('item_id', 'latest')}",
                    enable_found_actions=True,
                )
                render_lost_submission_reject_all(core, latest_result)

elif st.session_state["user_view"] == "found_upload":
    st.subheader("捡到东西的人：登记招领物，并检索失物区")
    st.write("上传拾到物品的照片并填写存放地点。提交后，这条记录会进入“招领区”，同时从“失物区”召回可能的失主登记。")

    form_cols = st.columns([1, 1])
    with form_cols[0]:
        found_images = st.file_uploader(
            "拾到物品照片（支持多选）",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="found_images",
        )
        found_location = st.selectbox("存放地点", location_options(include_any=False), key="found_location")
        operator_name = st.text_input(
            "登记人",
            value=current_user.get("account_id", ""),
            placeholder="例如：张同学 / 保卫处值班员",
            key="found_operator_name",
            disabled=True,
        )
        operator_contact = st.text_input(
            "登记人联系方式",
            value=current_user.get("phone", ""),
            placeholder="例如：手机号后四位 / 微信 / 工号",
            key="found_operator_contact",
        )
        found_note = st.text_input("拾物补充描述", placeholder="例如：杯身有熊猫贴纸、耳机盒有划痕")
        found_top_k = 5
        found_use_agent = st.checkbox("启用 DeepSeek 智能通知判断", value=True, key="found_use_agent")
        submit_found = st.button("确认登记", type="primary", use_container_width=True, disabled=not found_images)
    with form_cols[1]:
        if found_images:
            preview_cols = st.columns(min(2, len(found_images)))
            for index, uploaded_file in enumerate(found_images[:4]):
                with preview_cols[index % len(preview_cols)]:
                    st.image(uploaded_file, caption=uploaded_file.name, use_column_width=True)
            if len(found_images) > 4:
                st.caption(f"另有 {len(found_images) - 4} 张图片待处理。")
        else:
            st.info("请先上传拾到物品的图片。")

    if submit_found and found_images:
        rows = []
        latest_result = None
        with st.spinner("正在登记招领物，系统后台会自动检索并判断是否需要通知失主..."):
            progress = st.progress(0) if len(found_images) > 1 else None
            for index, uploaded_file in enumerate(found_images, start=1):
                try:
                    image_path = save_uploaded_file_unique(uploaded_file)
                    result = core.report_found_item(
                        image_path=image_path,
                        location=found_location,
                        reporter_note=found_note.strip(),
                        operator_name=current_user.get("account_id", ""),
                        operator_contact=operator_contact.strip(),
                        top_k=found_top_k,
                        use_agent=found_use_agent,
                    )
                    latest_result = result
                    st.session_state["selected_upload_matches"] = {
                        "item_id": result["item_id"],
                        "record_type": "found",
                    }
                    rows.append(
                        {
                            "文件": uploaded_file.name,
                            "状态": "成功",
                            "编号": result["item_id"],
                            "失败原因": "",
                        }
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "文件": uploaded_file.name,
                            "状态": "失败",
                            "编号": "",
                            "失败原因": str(exc),
                        }
                    )
                if progress:
                    progress.progress(index / len(found_images))
        render_batch_rows(rows)
        if latest_result:
            decision = latest_result.get("agent_decision") or {}
            sent_count = len(decision.get("sent_notifications") or [])
            if sent_count:
                st.success(f"后台已完成智能匹配，并向疑似失主生成 {sent_count} 条站内通知。")
            else:
                st.info("招领登记已完成。系统会继续在后台等待合适的失主登记。")

elif st.session_state["user_view"] == "found_library":
    render_found_library(core, key_prefix="user_found_library", current_user=current_user)

elif st.session_state["user_view"] == "lost_library":
    render_lost_library(core, key_prefix="user_lost_library")

elif st.session_state["user_view"] == "my_uploads":
    render_my_uploads(core, current_user)

elif st.session_state["user_view"] == "notifications":
    render_notification_inbox(core, current_user)

elif st.session_state["user_view"] == "notification_detail":
    render_notification_detail_page(core, current_user)
