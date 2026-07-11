"""
ai_services.py
Qwen-VL 结构化提取、证件脱敏路径与 DeepSeek 精排裁决。
"""

from __future__ import annotations

import json
import base64
import mimetypes
import os
import re
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from PIL import Image

load_dotenv()


ORDINARY_DEFAULTS = {
    "is_identity_document": False,
    "category": "未知物品",
    "subcategory": "未知",
    "color": "未知",
    "material": "未知",
    "size": "未知",
    "condition": "未知",
    "brand": "未知",
    "logo": "未识别",
    "text_visible": "无",
    "distinctive_marks": "无",
    "shape": "未知",
    "shape_profile": "未知",
    "main_object": "未知",
    "object_parts": [],
    "interface_type": "无",
    "pair_status": "未知",
    "accessories": "无",
    "damage_or_wear": "未识别",
    "location_hint": "未知",
    "search_keywords": [],
    "fine_grained_signature": "暂无细粒度特征",
    "appearance": "暂无外观描述",
}

FEATURE_SCHEMA_GUIDE = """
请严格按以下稳定口径提取字段，found/lost 两种场景必须使用同一套视觉描述标准：
- category：物品大类，优先用常见校园失物类别，如电子产品、手表、耳机、水杯、钥匙、证件、书本、雨伞、书包、钱包、文具；不确定才写未知物品。
- subcategory：细分类，必须尽量具体，例如充电线、数据线、充电器、无线耳机、有线耳机、耳机盒、手机、电子手表、眼镜盒、水杯、钥匙串；不要只写电子产品。
- main_object：更具体的主体名，如运动电子手表、白色无线耳机、蓝色保温杯。
- color：只写物品本体主色和关键辅色，例如白色和金色；不要写背景颜色。
- material：只写物品本体可见材质，例如塑料、硅胶、金属、皮革、布料；不确定写未知。
- shape：只写物品本体形状/结构，例如圆形表盘、椭圆形耳机盒、长柄耳机。
- shape_profile：更细的形态轮廓，例如细长线缆、方形充电头、入耳式耳机、椭圆耳机盒、圆形表盘。
- object_parts：数组，列出物品可见关键部件，例如线缆、USB-C接口、插头、耳塞、耳柄、充电盒、表带。
- interface_type：接口/连接类型，例如 USB-C、Lightning、3.5mm、无线、无、未知；看不清写未知。
- pair_status：成套状态，例如单只、一对、带盒、单独线缆、单独充电头、未知。
- 细分类强约束：如果主体是线状可弯曲物，优先判为充电线/数据线；如果主体是插墙或方块电源，判为充电器/充电头；如果主体是带耳塞或耳柄的入耳/半入耳设备，判为无线耳机/有线耳机；如果主体是开合盒体且用于收纳耳机，判为耳机盒。不要因为颜色相同而把这些物品混为同一类。
- 电子配件区分：充电线的关键部件通常是线缆、接口、接头；无线耳机的关键部件通常是耳塞、耳柄、出音孔、触控柄；耳机盒的关键部件通常是盒体、开合盖、铰链、充电口；充电器的关键部件通常是插脚、充电头、USB接口。
- brand/logo：只写物品上可见品牌、logo 或字样；不要猜测。
- text_visible：只写物品本体上可见文字、数字、符号，例如 AIKE、SPORT、WATER RESIST、14:03。
- distinctive_marks：写最适合人工核验的稳定细节，避免泛泛而谈。
- damage_or_wear：只写可见划痕、磨损、污渍、缺口；无明显痕迹写无。
- accessories：只写物品本身附带部件，例如表带、耳塞、挂绳、钥匙圈。
- location_hint：只有图片中出现明确地点文字/标志时填写，否则写未知。
- search_keywords：输出 6-10 个中文短关键词，使用同义词覆盖检索，如手表、电子表、运动表、白色、金色、硅胶表带。
- fine_grained_signature：用一句非常具体的区分性签名，必须包含细分类和 2-4 个稳定结构细节，例如“白色无线耳机，长耳柄，入耳硅胶耳塞，出音孔可见”。不要写背景。
- appearance：用 1 句中文，按固定顺序描述：细分类/主体 + 颜色 + 形状结构 + 可见文字/品牌 + 特征/磨损。只描述物品本体，不要描述拍摄环境。
""".strip()


def clean_json_response(raw_text: str) -> dict:
    if not raw_text or not raw_text.strip():
        raise ValueError("模型返回内容为空，无法解析")
    text = raw_text.strip()
    for candidate in [
        text,
        re.sub(r"```(?:json)?\s*\n?(.*?)\n?\s*```", r"\1", text, flags=re.DOTALL).strip(),
    ]:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        return json.loads(brace_match.group(0))
    raise ValueError(f"无法从模型输出中解析 JSON：{raw_text[:300]}")


def _dominant_color_name(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB").resize((32, 32))
    pixels = list(image.getdata())
    r, g, b = tuple(sum(channel) / len(pixels) for channel in zip(*pixels))
    if max(r, g, b) < 55:
        return "黑色"
    if min(r, g, b) > 205:
        return "白色"
    if r > g * 1.25 and r > b * 1.25:
        return "红色"
    if b > r * 1.2 and b > g * 1.1:
        return "蓝色"
    if g > r * 1.2 and g > b * 1.15:
        return "绿色"
    if r > 160 and g > 130 and b < 100:
        return "黄色"
    return "混合色"


def _guess_category(text: str) -> str:
    for keyword, category in {
        "充电线": "电子配件",
        "数据线": "电子配件",
        "充电器": "电子配件",
        "充电头": "电子配件",
        "校园卡": "校园卡",
        "学生证": "学生证",
        "身份证": "身份证",
        "水杯": "水杯",
        "保温杯": "水杯",
        "杯": "水杯",
        "耳机": "耳机",
        "钥匙": "钥匙",
        "雨伞": "雨伞",
        "伞": "雨伞",
        "书包": "书包",
        "包": "书包",
        "书": "书本",
        "手机": "手机",
    }.items():
        if keyword in text:
            return category
    return "未知物品"


def _guess_subcategory(text: str) -> str:
    lowered = text.lower()
    for keyword, subcategory in {
        "耳机盒": "耳机盒",
        "充电盒": "耳机盒",
        "耳机仓": "耳机盒",
        "无线耳机": "无线耳机",
        "蓝牙耳机": "无线耳机",
        "airpods": "无线耳机",
        "耳塞": "无线耳机",
        "耳柄": "无线耳机",
        "有线耳机": "有线耳机",
        "耳机线": "有线耳机",
        "充电线": "充电线",
        "数据线": "充电线",
        "type-c": "充电线",
        "usb-c": "充电线",
        "lightning": "充电线",
        "充电器": "充电器",
        "充电头": "充电器",
        "适配器": "充电器",
        "手表": "手表",
        "水杯": "水杯",
        "保温杯": "水杯",
        "钥匙": "钥匙",
    }.items():
        if keyword in lowered or keyword in text:
            return subcategory
    return "未知"


def is_identity_document_hint(image_path: str, note: str = "") -> bool:
    text = f"{Path(image_path).name} {note}".lower()
    return any(word in text for word in ["身份证", "校园卡", "学生证", "idcard", "student_card", "card"])


def _mask_identity_from_text(text: str) -> dict:
    student_id = re.search(r"\d{8,14}", text)
    name = re.search(r"([\u4e00-\u9fff])[\u4e00-\u9fff]{1,3}", text)
    sid = student_id.group(0) if student_id else ""
    return {
        "has_personal_info": bool(student_id or name),
        "student_id_partial": f"{sid[:4]}****{sid[-4:]}" if len(sid) >= 8 else "",
        "name_masked": f"{name.group(1)}*" if name else "",
        "school": "未知学院",
    }


def _strip_background_details(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return text
    blocked_keywords = [
        "背景",
        "桌面",
        "台面",
        "地面",
        "床单",
        "被子",
        "布料",
        "纸张上",
        "放置在",
        "摆放在",
        "置于",
        "拍摄",
        "光线",
        "阴影",
        "水印",
        "Photographer",
        "photographer",
    ]
    parts = re.split(r"(?<=[。！？；;])", text)
    kept = [part for part in parts if not any(keyword in part for keyword in blocked_keywords)]
    cleaned = "".join(kept).strip(" ，。；;")
    return cleaned or text


def _sanitize_object_features(result: dict) -> dict:
    cleaned = dict(result)
    for key in ["appearance", "distinctive_marks", "location_hint"]:
        if key in cleaned:
            cleaned[key] = _strip_background_details(cleaned[key])
    for key in [
        "category",
        "subcategory",
        "main_object",
        "color",
        "material",
        "size",
        "condition",
        "brand",
        "logo",
        "text_visible",
        "shape",
        "shape_profile",
        "interface_type",
        "pair_status",
        "accessories",
        "damage_or_wear",
        "fine_grained_signature",
    ]:
        if isinstance(cleaned.get(key), str):
            value = re.sub(r"\s+", " ", cleaned[key]).strip(" '\"，,；;。")
            cleaned[key] = value or ORDINARY_DEFAULTS.get(key, "未知")
    parts = cleaned.get("object_parts", [])
    if isinstance(parts, str):
        parts = re.split(r"[，,、\s]+", parts)
    if isinstance(parts, list):
        deduped_parts = []
        for part in parts:
            text = str(part).strip(" '\"，,；;。")
            if text and text not in deduped_parts and text not in {"未知", "无"}:
                deduped_parts.append(text)
        cleaned["object_parts"] = deduped_parts[:8]
    keywords = cleaned.get("search_keywords", [])
    if isinstance(keywords, str):
        keywords = re.split(r"[，,、\s]+", keywords)
    if isinstance(keywords, list):
        deduped = []
        for keyword in keywords:
            text = str(keyword).strip(" '\"，,；;。")
            if text and text not in deduped and text not in {"未知", "无"}:
                deduped.append(text)
        cleaned["search_keywords"] = deduped[:10]
    return cleaned


class VisionExtractor:
    """普通物品走 Qwen-VL；证件类走本地脱敏路径，不上传云端。"""

    def __init__(self, api_key: Optional[str] = None, prefer_remote: bool = True):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.prefer_remote = prefer_remote and bool(self.api_key)
        self.model_name = os.getenv("QWEN_VL_MODEL", "qwen-vl-plus")
        self.base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.last_source = "not_called"
        self.last_error = ""
        self.system_prompt = (
            "你是一个校园失物招领系统的图像转文字识别小助手，负责把物品图片转换为稳定、可检索、可复核的结构化 JSON。"
            "你的目标是让同一个物品无论出现在拾取登记还是丢失登记中，都生成风格一致、字段含义一致、细节粒度一致的描述。"
            "你的输出会直接进入图文向量库，所以细分类和结构特征比泛泛的大类更重要。"
            "先判断是否证件类物品；普通物品只输出严格 JSON，不要输出解释、Markdown 或代码块。"
            "JSON 字段必须包含：is_identity_document, category, subcategory, main_object, color, material, size, "
            "shape, shape_profile, condition, brand, logo, text_visible, distinctive_marks, object_parts, "
            "interface_type, pair_status, accessories, damage_or_wear, location_hint, search_keywords, "
            "fine_grained_signature, appearance。"
            "只描述图片中的物品本体，不要描述背景、桌面、布料、地面、拍摄光线、水印或摆放环境。"
            f"{FEATURE_SCHEMA_GUIDE}"
        )

    @property
    def mode(self) -> str:
        return "qwen-vl" if self.prefer_remote else "local-vision-fallback"

    def extract_item_info(self, image_path: str, reporter_note: str = "", record_type: str = "unknown") -> dict:
        self.last_source = "not_called"
        self.last_error = ""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片文件不存在: {image_path}")
        if is_identity_document_hint(image_path, reporter_note):
            result = self.extract_identity_document_locally(image_path, reporter_note)
            result["_vlm_source"] = "identity_local"
            self.last_source = "identity_local"
            return result
        if self.prefer_remote:
            errors = []
            try:
                result = self._extract_with_openai_compatible(image_path, reporter_note, record_type)
                result["_vlm_source"] = "qwen_vl_openai_compatible"
                self.last_source = result["_vlm_source"]
                return result
            except Exception as exc:
                errors.append(f"openai_compatible={exc}")
                print(f"[Qwen-VL] OpenAI兼容接口调用失败：{exc}", flush=True)
            try:
                result = self._extract_with_qwen_sdk(image_path, reporter_note, record_type)
                result["_vlm_source"] = "qwen_vl_dashscope_sdk"
                self.last_source = result["_vlm_source"]
                return result
            except Exception as exc:
                errors.append(f"dashscope_sdk={exc}")
                print(f"[Qwen-VL] DashScope SDK调用失败：{exc}", flush=True)
            self.last_error = "；".join(errors)
        elif not self.api_key:
            self.last_error = "未配置 DASHSCOPE_API_KEY"
        fallback = self._extract_locally(image_path, reporter_note)
        fallback["_vlm_source"] = "local_fallback"
        fallback["_vlm_error"] = self.last_error
        self.last_source = "local_fallback"
        print(f"[Qwen-VL] 使用本地兜底：{self.last_error or '未启用远程视觉模型'}", flush=True)
        return fallback

    def extract_identity_document_locally(self, image_path: str, note: str = "") -> dict:
        identity = _mask_identity_from_text(f"{Path(image_path).stem} {note}")
        return {
            "is_identity_document": True,
            "category": _guess_category(f"{Path(image_path).name} {note}") if note else "证件",
            "identity": identity,
            "appearance": "证件类物品已走本地脱敏路径，敏感信息不进入向量库。",
        }

    def _build_user_prompt(self, reporter_note: str = "", record_type: str = "unknown") -> str:
        scenario = {"found": "拾取登记", "lost": "寻物登记"}.get(record_type, "通用登记")
        note = reporter_note.strip() if reporter_note else "无"
        return (
            f"当前场景：{scenario}。请注意：场景只影响 record_type，不影响视觉字段的写法和粒度。"
            "请按同一套字段标准提取图片中物品本体的可检索细节。"
            f"用户补充描述：{note}。"
            "用户补充可以辅助判断类别和细节，但不要覆盖图片中清晰可见的信息。"
            "请优先判断 subcategory，并围绕 subcategory 输出 object_parts、shape_profile、interface_type、pair_status 和 fine_grained_signature。"
            "如果图片主体是白色电子类物品，也必须继续区分它到底是充电线、充电器、无线耳机、有线耳机、耳机盒还是其他电子产品。"
            "只输出严格 JSON，不要解释。"
        )

    def _extract_with_openai_compatible(self, image_path: str, reporter_note: str = "", record_type: str = "unknown") -> dict:
        from openai import OpenAI

        mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
        with open(image_path, "rb") as file:
            image_b64 = base64.b64encode(file.read()).decode("utf-8")
        client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=60, max_retries=1)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
                        {"type": "text", "text": self._build_user_prompt(reporter_note, record_type)},
                    ],
                },
            ],
            temperature=0,
        )
        raw = response.choices[0].message.content
        return self._normalize_features(clean_json_response(raw))

    def _extract_with_qwen_sdk(self, image_path: str, reporter_note: str = "", record_type: str = "unknown") -> dict:
        import dashscope
        from dashscope import MultiModalConversation

        dashscope.api_key = self.api_key
        messages = [
            {"role": "system", "content": [{"text": self.system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"image": f"file://{os.path.abspath(image_path)}"},
                    {"text": self._build_user_prompt(reporter_note, record_type)},
                ],
            },
        ]
        response = MultiModalConversation.call(model=self.model_name, messages=messages)
        if response.status_code != 200:
            raise RuntimeError(response.message)
        raw = response.output.choices[0].message.content[0]["text"]
        return self._normalize_features(clean_json_response(raw))

    def _extract_locally(self, image_path: str, note: str = "") -> dict:
        color = _dominant_color_name(image_path)
        hint_text = f"{Path(image_path).stem} {note}"
        category = _guess_category(hint_text)
        subcategory = _guess_subcategory(hint_text)
        object_parts = []
        if subcategory == "充电线":
            object_parts = ["线缆", "接口"]
        elif subcategory == "充电器":
            object_parts = ["插头", "充电头"]
        elif subcategory in {"无线耳机", "有线耳机"}:
            object_parts = ["耳塞", "耳机主体"]
        elif subcategory == "耳机盒":
            object_parts = ["盒体", "开合盖"]
        data = {
            **ORDINARY_DEFAULTS,
            "category": category,
            "subcategory": subcategory,
            "color": color,
            "object_parts": object_parts,
            "shape_profile": "细长线缆" if subcategory == "充电线" else "未知",
            "interface_type": "未知",
            "fine_grained_signature": f"{color}{subcategory if subcategory != '未知' else category}",
            "distinctive_marks": note or "无",
            "appearance": f"{color}{subcategory if subcategory != '未知' else category}，由本地视觉兜底根据图片颜色、文件名和拾物备注生成。",
        }
        return self._normalize_features(data)

    def _normalize_features(self, result: dict) -> dict:
        if result.get("is_identity_document"):
            return {
                "is_identity_document": True,
                "category": result.get("category", "证件"),
                "identity": result.get("identity", {}),
                "appearance": result.get("appearance", "证件类物品已脱敏。"),
            }
        merged = {**ORDINARY_DEFAULTS, **{key: value for key, value in result.items() if value not in [None, ""]}}
        return _sanitize_object_features(merged)


class MatchingAgent:
    """DeepSeek 精排裁决，决定是否通知用户以及通知形式。"""

    def __init__(self, api_key: Optional[str] = None, prefer_remote: bool = True):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.prefer_remote = prefer_remote and bool(self.api_key)
        self.model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.system_prompt = (
            "你是校园失物招领系统的智能裁决与通知 Agent。"
            "你会收到一条新登记记录和向量召回 TopK 候选。你的任务不是机械相信向量分数，"
            "而是结合物品细分类、颜色、结构部件、文字/logo、地点、时间、状态、用户描述，判断是否应该通知用户。"
            "你必须优先避免误通知。只有证据充分时才通知。"
            "如果最高候选高度可信且明显优于其他候选，调用 notify_single_candidate。"
            "如果有 2-5 个候选都比较相似、无法唯一确认，调用 notify_multiple_candidates，并由你决定应该通知几个候选。"
            "【近分候选规则】如果 Top1 与 Top2/Top3 的向量分差小于等于 0.04，且它们属于同一细类或结构高度相近，"
            "即使 Top1 分数最高，也不要单物品通知，应调用 notify_multiple_candidates，让用户同时核验多个候选。"
            "例如多个白色充电器/充电线候选都在 0.90 以上且分差很小，应多物品推选。"
            "【候选数量规则】多候选通知不是固定发送全部 TopK。你要只选择真正可能是同一件物品的候选 ID："
            "通常 2-3 个，最多 5 个；遇到明显分数断崖、类别/颜色/接口/结构冲突、描述不符的候选要排除。"
            "如果 Top1-Top4 都很接近且都是合理候选，可以通知 4 个；如果 Top5 明显断崖或类型不符，不要通知 Top5。"
            "如果证据不足、类别冲突、颜色/结构明显不符、候选太弱，调用 keep_manual_review。"
            "【否定语义规则】如果失主描述中出现不是、没有、不带、不像、排除，"
            "请将否定词后的属性作为强排除条件；包含该属性的候选物 confidence_score 强制 20 以下，"
            "不得触发通知。"
            "【关键约束】充电线、充电器、无线耳机、耳机盒虽然都可能是电子配件，但结构差异很大，不得仅因颜色相似就判定匹配。"
            "【输出格式约束】function calling 的字符串参数中不要使用未转义的英文双引号；"
            "如果要引用按钮文案，请使用中文引号，例如“已取回”，避免生成无法解析的 JSON 参数。"
            "输出必须通过 function calling；如果环境不支持 function calling，则输出等价严格 JSON。"
            "请仅输出合法的 JSON，不要输出 Markdown 代码块、解释文字或其他内容。"
        )
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "notify_single_candidate",
                    "description": "通知用户存在一个高度可信的单一匹配候选。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "matched_item_id": {"type": "string"},
                            "confidence_score": {"type": "integer", "minimum": 0, "maximum": 100},
                            "message_title": {"type": "string"},
                            "message_body": {"type": "string"},
                            "pickup_guide": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                        "required": [
                            "matched_item_id",
                            "confidence_score",
                            "message_title",
                            "message_body",
                            "pickup_guide",
                            "reason",
                        ],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "notify_multiple_candidates",
                    "description": "通知用户存在多个可能候选，需要用户查看图片或到现场核验。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "candidate_item_ids": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 5},
                            "candidate_count": {"type": "integer", "minimum": 2, "maximum": 5},
                            "confidence_score": {"type": "integer", "minimum": 0, "maximum": 100},
                            "message_title": {"type": "string"},
                            "message_body": {"type": "string"},
                            "pickup_guide": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                        "required": [
                            "candidate_item_ids",
                            "candidate_count",
                            "confidence_score",
                            "message_title",
                            "message_body",
                            "pickup_guide",
                            "reason",
                        ],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "keep_manual_review",
                    "description": "不通知用户，保留人工复核或继续等待。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "confidence_score": {"type": "integer", "minimum": 0, "maximum": 100},
                            "reason": {"type": "string"},
                            "pickup_guide": {"type": "string"},
                        },
                        "required": ["confidence_score", "reason", "pickup_guide"],
                    },
                },
            },
        ]

    @property
    def mode(self) -> str:
        return "deepseek" if self.prefer_remote else "local-agent-fallback"

    def evaluate_candidates(self, loser_query: str, candidates: list[dict], direction: str = "lost_to_found") -> dict:
        if not candidates:
            return self._empty_result("候选列表为空")
        best_score = float(candidates[0].get("score", 0))
        if best_score < 0.35 and self._keyword_hits(loser_query, candidates[0]) < 2:
            return self._empty_result("最高向量相似度低于 0.35，跳过 DeepSeek")
        notification_pool = self._notification_candidate_pool(candidates)
        if best_score > 0.92 and self._category_hit(loser_query, candidates[0]) and len(notification_pool) < 2:
            return self._fast_match(candidates[0])
        if self.prefer_remote:
            try:
                return self._evaluate_with_deepseek(loser_query, candidates, direction)
            except Exception as exc:
                print(f"[DeepSeek] 调用失败，使用本地精排：{exc}", flush=True)
                self.prefer_remote = False
        return self._evaluate_locally(loser_query, candidates)

    def _highlight_negations(self, query: str) -> str:
        for pattern in ["不是", "没有", "不带", "不像", "排除"]:
            query = query.replace(pattern, f"【否定】{pattern}")
        return query

    def _evaluate_with_deepseek(self, loser_query: str, candidates: list[dict], direction: str) -> dict:
        from openai import OpenAI

        notification_pool = self._notification_candidate_pool(candidates)
        close_hint = "无明显多候选通知池"
        if notification_pool:
            close_hint = json.dumps(
                [
                    {
                        "item_id": item.get("item_id"),
                        "score": item.get("score"),
                        "category": item.get("metadata", {}).get("category"),
                        "appearance": item.get("metadata", {}).get("appearance", "")[:80],
                    }
                    for item in notification_pool
                ],
                ensure_ascii=False,
                indent=2,
            )
        score_gaps = []
        for index in range(min(len(candidates) - 1, 5)):
            current_score = float(candidates[index].get("score", 0))
            next_score = float(candidates[index + 1].get("score", 0))
            score_gaps.append(
                {
                    "from": f"Top{index + 1}",
                    "to": f"Top{index + 2}",
                    "gap": round(current_score - next_score, 4),
                }
            )

        client = OpenAI(
            api_key=self.api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            timeout=30,
            max_retries=1,
        )
        user_message = (
            f"【匹配方向】\n{direction}\n"
            "lost_to_found 表示失物登记要匹配招领区候选；found_to_lost 表示招领登记要匹配失物区候选。\n\n"
            f"【新登记记录 JSON】\n{self._highlight_negations(loser_query)}\n\n"
            f"【TopK 候选物 JSON】\n{json.dumps(candidates, ensure_ascii=False, indent=2)}\n\n"
            f"【建议交给你判断的多候选通知池】\n{close_hint}\n\n"
            f"【相邻分数差】\n{json.dumps(score_gaps, ensure_ascii=False, indent=2)}\n\n"
            "请你决定本次到底通知 0 个、1 个，还是多个候选。"
            "如果调用 notify_multiple_candidates，candidate_item_ids 必须只包含你建议发给用户的候选，"
            "candidate_count 必须等于 candidate_item_ids 数量。不要为了凑数加入明显断崖或不相似的候选。\n"
            "请根据证据选择一个函数调用。通知文案要克制，不要承诺一定匹配，提醒用户核验细节。"
            "如果没有触发函数调用，请仅输出合法的 JSON，不要输出 Markdown 代码块、解释文字或其他内容。"
        )
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": self.system_prompt}, {"role": "user", "content": user_message}],
            tools=self.tools,
            tool_choice="auto",
            temperature=0.2,
            max_tokens=512,
        )
        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None) or []
        if tool_calls:
            tool_call = tool_calls[0]
            fn = tool_call.function
            args = self._parse_deepseek_tool_args(fn.arguments or "{}", fn.name)
            return self._normalize_tool_decision(fn.name, args, candidates)
        return self._normalize_decision(clean_json_response(message.content or ""), candidates)

    def _evaluate_locally(self, loser_query: str, candidates: list[dict]) -> dict:
        notification_pool = self._notification_candidate_pool(candidates)
        if len(notification_pool) >= 2:
            return self._multiple_match(notification_pool)

        best = candidates[0]
        meta = best.get("metadata", {})
        combined = json.dumps(meta, ensure_ascii=False)
        hits = self._keyword_hits(loser_query, best)
        confidence = max(0, min(100, round(float(best.get("score", 0)) * 60 + hits * 16)))
        if self._has_negation_conflict(loser_query, combined):
            confidence = min(confidence, 20)
        action = "notify_user" if confidence >= 72 else "require_manual_review"
        notification_type = "single" if action == "notify_user" else "none"
        return {
            "confidence_score": confidence,
            "matched_item_id": best.get("item_id"),
            "candidate_item_ids": [best.get("item_id")] if action == "notify_user" else [],
            "decision": "matched" if action == "notify_user" else "manual_review",
            "action": action,
            "notification_type": notification_type,
            "message_title": "疑似找到匹配物品" if action == "notify_user" else "",
            "message_body": "系统发现一条相似度较高的候选记录，请核验图片和物品细节。" if action == "notify_user" else "",
            "pickup_guide": (
                f"疑似匹配成功，请前往{meta.get('location', '对应驿站')}核验领取，并说明物品细节。"
                if action == "notify_user"
                else "相似度不足，建议继续人工确认或扩大搜索范围。"
            ),
            "reason": "本地规则根据向量相似度、关键词重合度和否定语义生成裁决",
        }

    def _notification_candidate_pool(self, candidates: list[dict], max_items: int = 5) -> list[dict]:
        if len(candidates) < 2:
            return []
        top_score = float(candidates[0].get("score", 0))
        pool = [candidates[0]]
        previous_score = top_score
        for candidate in candidates[:max_items]:
            if candidate is candidates[0]:
                continue
            score = float(candidate.get("score", 0))
            gap_from_previous = previous_score - score
            if score < 0.70 or top_score - score > 0.08 or gap_from_previous > 0.12:
                break
            pool.append(candidate)
            previous_score = score
        return pool if len(pool) >= 2 else []

    def _multiple_match(self, candidates: list[dict]) -> dict:
        candidate_ids = [candidate.get("item_id") for candidate in candidates if candidate.get("item_id")]
        best_score = float(candidates[0].get("score", 0)) if candidates else 0
        return {
            "confidence_score": max(70, min(92, round(best_score * 100))),
            "matched_item_id": candidate_ids[0] if candidate_ids else None,
            "candidate_item_ids": candidate_ids[:5],
            "candidate_count": min(len(candidate_ids), 5),
            "decision": "multiple_candidates",
            "action": "notify_user",
            "notification_type": "multiple",
            "message_title": "发现多个相似候选物品",
            "message_body": "系统发现多条高度接近的候选记录，请逐张核验图片、接口、磨损、文字标识等细节。",
            "pickup_guide": "建议同时查看候选图片或到对应地点人工核验，不要仅凭颜色和大类判断。",
            "reason": "Top 候选向量分差很小，无法唯一确认，按多物品候选推送。",
        }

    def _keyword_hits(self, query: str, candidate: dict) -> int:
        combined = json.dumps(candidate.get("metadata", {}), ensure_ascii=False)
        return sum(
            1
            for key in ["蓝色", "黑色", "白色", "红色", "水杯", "保温杯", "耳机", "钥匙", "校园卡", "贴纸", "熊猫", "吊牌"]
            if key in query and key in combined
        )

    def _has_negation_conflict(self, query: str, combined: str) -> bool:
        for neg in ["不是", "没有", "不带", "不像", "排除"]:
            if neg in query:
                tail = query.split(neg, 1)[1][:8]
                if any(ch and ch in combined for ch in re.findall(r"[\u4e00-\u9fff]{2,4}", tail)):
                    return True
        return False

    def _category_hit(self, query: str, candidate: dict) -> bool:
        category = str(candidate.get("metadata", {}).get("category", ""))
        return bool(category and category in query)

    def _fast_match(self, candidate: dict) -> dict:
        meta = candidate.get("metadata", {})
        return {
            "confidence_score": 96,
            "matched_item_id": candidate.get("item_id"),
            "candidate_item_ids": [candidate.get("item_id")],
            "decision": "matched",
            "action": "notify_user",
            "notification_type": "single",
            "message_title": "疑似找到匹配物品",
            "message_body": "系统发现一条高度相似的候选记录，请尽快核验图片和物品细节。",
            "pickup_guide": f"高度匹配，请前往{meta.get('location', '对应驿站')}核验领取。",
            "reason": "向量相似度超过 0.92 且类别命中，快速通过",
        }

    def _empty_result(self, reason: str) -> dict:
        return {
            "confidence_score": 0,
            "matched_item_id": None,
            "candidate_item_ids": [],
            "decision": "not_found",
            "action": "keep_waiting",
            "notification_type": "none",
            "message_title": "",
            "message_body": "",
            "pickup_guide": "暂未找到高相似候选物，建议扩大时间或地点范围后重试。",
            "reason": reason,
        }

    def _parse_deepseek_tool_args(self, raw_args: str | dict, function_name: str) -> dict:
        if isinstance(raw_args, dict):
            return raw_args
        raw_text = (raw_args or "{}").strip()
        for parser in (json.loads, clean_json_response):
            try:
                parsed = parser(raw_text)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

        print(
            f"[DeepSeek] function 参数不是合法 JSON，尝试字段级兜底解析："
            f"fn={function_name}, raw={raw_text[:500]}",
            flush=True,
        )
        return self._recover_tool_args_from_text(raw_text, function_name)

    def _recover_tool_args_from_text(self, raw_text: str, function_name: str) -> dict:
        recovered: dict = {}

        def pick_string(key: str) -> str:
            match = re.search(rf'"{re.escape(key)}"\s*:\s*"([^"]*)"', raw_text)
            return match.group(1).strip() if match else ""

        def pick_int(key: str) -> int:
            match = re.search(rf'"{re.escape(key)}"\s*:\s*"?(\d+)"?', raw_text)
            return int(match.group(1)) if match else 0

        def pick_list(key: str) -> list[str]:
            match = re.search(rf'"{re.escape(key)}"\s*:\s*\[(.*?)\]', raw_text, re.DOTALL)
            if not match:
                return []
            return [item.strip() for item in re.findall(r'"([^"]+)"', match.group(1)) if item.strip()]

        if function_name == "notify_single_candidate":
            recovered["matched_item_id"] = pick_string("matched_item_id")
        elif function_name == "notify_multiple_candidates":
            recovered["candidate_item_ids"] = pick_list("candidate_item_ids")
            recovered["candidate_count"] = pick_int("candidate_count") or len(recovered["candidate_item_ids"])

        recovered["confidence_score"] = pick_int("confidence_score")
        recovered["message_title"] = pick_string("message_title")
        recovered["message_body"] = pick_string("message_body")
        recovered["pickup_guide"] = pick_string("pickup_guide")
        recovered["reason"] = pick_string("reason") or "DeepSeek 返回了不完整 JSON，系统已按可恢复字段继续处理。"
        return recovered

    def _normalize_tool_decision(self, function_name: str, args: dict, candidates: list[dict]) -> dict:
        if function_name == "notify_single_candidate":
            result = {
                **args,
                "decision": "matched",
                "action": "notify_user",
                "notification_type": "single",
                "candidate_item_ids": [args.get("matched_item_id")] if args.get("matched_item_id") else [],
            }
        elif function_name == "notify_multiple_candidates":
            candidate_ids = self._valid_candidate_ids(args.get("candidate_item_ids", []), candidates)
            if len(candidate_ids) < 2:
                candidate_ids = [
                    candidate.get("item_id")
                    for candidate in self._notification_candidate_pool(candidates)
                    if candidate.get("item_id")
                ][:5]
            result = {
                **args,
                "candidate_item_ids": candidate_ids,
                "candidate_count": len(candidate_ids),
                "matched_item_id": candidate_ids[0] if candidate_ids else None,
                "decision": "multiple_candidates" if len(candidate_ids) >= 2 else "manual_review",
                "action": "notify_user" if len(candidate_ids) >= 2 else "require_manual_review",
                "notification_type": "multiple" if len(candidate_ids) >= 2 else "none",
            }
        else:
            result = {
                **args,
                "matched_item_id": None,
                "candidate_item_ids": [],
                "decision": "manual_review",
                "action": "require_manual_review",
                "notification_type": "none",
                "message_title": "",
                "message_body": "",
            }
        return self._normalize_decision(result, candidates)

    def _normalize_decision(self, result: dict, candidates: list[dict]) -> dict:
        score = result.get("confidence_score", 0)
        if isinstance(score, str):
            found = re.search(r"\d+", score)
            score = int(found.group(0)) if found else 0
        score = max(0, min(100, int(score)))
        result["confidence_score"] = score
        result.setdefault("matched_item_id", candidates[0].get("item_id") if candidates and score >= 80 else None)
        if result.get("candidate_item_ids"):
            result["candidate_item_ids"] = self._valid_candidate_ids(result["candidate_item_ids"], candidates)
        else:
            result["candidate_item_ids"] = [result["matched_item_id"]] if result.get("matched_item_id") else []
        result["candidate_count"] = len(result.get("candidate_item_ids", []))
        result.setdefault("decision", "matched" if score >= 80 else "manual_review")
        result.setdefault("action", "notify_user" if score >= 80 else "require_manual_review")
        if result.get("notification_type") == "multiple" and result["candidate_count"] < 2:
            result["notification_type"] = "none"
            result["action"] = "require_manual_review"
            result["decision"] = "manual_review"
        result.setdefault("notification_type", "single" if result.get("action") == "notify_user" else "none")
        result.setdefault("message_title", "疑似找到匹配物品" if result.get("action") == "notify_user" else "")
        result.setdefault("message_body", "系统发现相似候选，请核验图片和物品细节。" if result.get("action") == "notify_user" else "")
        result.setdefault("pickup_guide", "请前往对应驿站人工核验领取。")
        result.setdefault("reason", "DeepSeek 精排结果")
        return result

    def _valid_candidate_ids(self, candidate_ids: list, candidates: list[dict]) -> list[str]:
        requested = [str(item_id) for item_id in candidate_ids if item_id]
        requested_set = set(requested)
        ordered = []
        for candidate in candidates:
            item_id = candidate.get("item_id")
            if item_id and item_id in requested_set and item_id not in ordered:
                ordered.append(item_id)
        return ordered[:5]
