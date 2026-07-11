"""
workflow.py
校园智能挂失系统 - 双向量库 RAG 编排层

当前跑通 RAG 部分，并可选接入 DeepSeek 智能裁决：
1. 拾取物品向量库 found_items：捡到东西的人上传图片/描述后入库
2. 寻物品向量库 lost_items：丢东西的人提交描述/图片后入库
3. 双向匹配：任一侧登记后，立即到另一侧向量库检索 TopK 候选

Agent 裁决会在开关启用时判断是否通知、通知单候选还是多候选。
"""

from __future__ import annotations

import shutil
import uuid
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from ai_services import MatchingAgent, VisionExtractor
from vector_db import ClipEncoder, VectorStoreManager, build_clip_search_string, fuse_vectors

load_dotenv()


FOUND_COLLECTION = "campus_found_items"
LOST_COLLECTION = "campus_lost_items"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _file_sha256(file_path: str) -> str:
    digest = hashlib.sha256()
    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _guess_category_from_text(text: str) -> str:
    mapping = {
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
        "手表": "手表",
        "钱包": "钱包",
        "充电器": "充电器",
    }
    for keyword, category in mapping.items():
        if keyword in text:
            return category
    return "未知物品"


def _guess_color_from_text(text: str) -> str:
    for color in ["黑色", "白色", "蓝色", "红色", "绿色", "黄色", "粉色", "紫色", "灰色", "银色"]:
        if color in text:
            return color
    return "未知"


def _vlm_runtime_metadata(vlm_json: Optional[dict], fallback_model: str = "text-only") -> dict:
    if not vlm_json:
        return {"vlm_model": fallback_model, "vlm_source": fallback_model, "vlm_error": ""}
    source = vlm_json.get("_vlm_source", "")
    error = vlm_json.get("_vlm_error", "")
    if source.startswith("qwen_vl"):
        model = os.getenv("QWEN_VL_MODEL", "qwen-vl-plus")
    elif source == "identity_local":
        model = "identity_local"
    elif source == "local_fallback":
        model = "local_fallback"
    else:
        model = os.getenv("QWEN_VL_MODEL", "qwen-vl-plus")
    return {"vlm_model": model, "vlm_source": source or model, "vlm_error": error}


class CampusSystemCore:
    """校园智能挂失系统核心控制器，维护寻物品库和拾取物品库。"""

    def __init__(self, data_dir: str = "./runtime_data"):
        self.data_dir = Path(data_dir)
        self.upload_dir = self.data_dir / "uploads"
        self.lost_upload_dir = self.data_dir / "lost_uploads"
        self.agent_log_path = self.data_dir / "agent_decision_logs.json"
        self.notification_log_path = self.data_dir / "notification_messages.json"
        self.resolution_log_path = self.data_dir / "resolution_logs.json"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.lost_upload_dir.mkdir(parents=True, exist_ok=True)

        shared_encoder = ClipEncoder()
        self.found_store = VectorStoreManager(
            str(self.data_dir / "found_items_db"),
            collection_name=FOUND_COLLECTION,
            encoder=shared_encoder,
        )
        self.lost_store = VectorStoreManager(
            str(self.data_dir / "lost_items_db"),
            collection_name=LOST_COLLECTION,
            encoder=shared_encoder,
        )
        self.vision_extractor = VisionExtractor()
        self.matching_agent = MatchingAgent()
        print(
            "[系统] 双库 RAG 初始化完成："
            f"found={self.found_store.get_collection_count()}, "
            f"lost={self.lost_store.get_collection_count()}, "
            f"Vision={self.vision_extractor.mode}, CLIP={shared_encoder.mode}"
        )

    def _rag_only_decision(self, reason: str = "未启用 DeepSeek 智能通知判断") -> dict:
        return {
            "decision": "rag_only",
            "action": "manual_review",
            "notification_type": "none",
            "matched_item_id": None,
            "candidate_item_ids": [],
            "confidence_score": 0,
            "message_title": "",
            "message_body": "",
            "pickup_guide": "当前只启用 RAG 召回，请人工核验 Top 候选。",
            "reason": reason,
        }

    def _evaluate_notification(
        self,
        query_record: dict,
        candidates: list[dict],
        enabled: bool,
        direction: str,
        record_id: str = "",
    ) -> dict:
        if not enabled:
            return self._rag_only_decision()
        query_text = json.dumps(query_record, ensure_ascii=False)
        decision = self.matching_agent.evaluate_candidates(query_text, candidates, direction=direction)
        sent_notifications = self._dispatch_agent_notification(record_id, query_record, candidates, decision, direction)
        if sent_notifications:
            decision["sent_notifications"] = sent_notifications
        self._record_agent_decision(
            record_id=record_id,
            direction=direction,
            query_record=query_record,
            candidates=candidates,
            decision=decision,
        )
        return decision

    def _load_notification_logs(self) -> list[dict]:
        if not self.notification_log_path.exists():
            return []
        try:
            data = json.loads(self.notification_log_path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def _save_notification_logs(self, logs: list[dict]) -> None:
        self.notification_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.notification_log_path.write_text(json.dumps(logs[-500:], ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_resolution_logs(self) -> list[dict]:
        if not self.resolution_log_path.exists():
            return []
        try:
            data = json.loads(self.resolution_log_path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def _save_resolution_logs(self, logs: list[dict]) -> None:
        self.resolution_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.resolution_log_path.write_text(json.dumps(logs[-1000:], ensure_ascii=False, indent=2), encoding="utf-8")

    def _dispatch_agent_notification(
        self,
        record_id: str,
        query_record: dict,
        candidates: list[dict],
        decision: dict,
        direction: str,
    ) -> list[dict]:
        if decision.get("action") != "notify_user":
            return []
        candidate_ids = set(decision.get("candidate_item_ids") or [])
        selected_candidates = [item for item in candidates if item.get("item_id") in candidate_ids]
        if not selected_candidates and decision.get("matched_item_id"):
            selected_candidates = [item for item in candidates if item.get("item_id") == decision.get("matched_item_id")]
        if not selected_candidates:
            return []

        messages = []
        if direction == "lost_to_found":
            recipient_name = query_record.get("operator_name") or "失主"
            recipient_contact = query_record.get("operator_contact") or query_record.get("contact") or "未填写"
            messages.append(
                self._build_notification_message(
                    record_id=record_id,
                    recipient_role="lost_owner",
                    recipient_name=recipient_name,
                    recipient_contact=recipient_contact,
                    related_item_ids=[item.get("item_id") for item in selected_candidates if item.get("item_id")],
                    pushed_candidates=self._candidate_message_snapshot(selected_candidates),
                    decision=decision,
                    direction=direction,
                )
            )
        else:
            for item in selected_candidates:
                metadata = item.get("metadata", {})
                pushed_found_snapshot = {
                    "rank": item.get("rank"),
                    "item_id": record_id,
                    "score": item.get("score"),
                    "distance": item.get("distance"),
                    "metadata": query_record,
                }
                messages.append(
                    self._build_notification_message(
                        record_id=record_id,
                        recipient_role="lost_owner",
                        recipient_name=metadata.get("operator_name") or "失主",
                        recipient_contact=metadata.get("operator_contact") or metadata.get("contact") or "未填写",
                        related_item_ids=[item.get("item_id")] if item.get("item_id") else [],
                        pushed_candidates=self._candidate_message_snapshot([pushed_found_snapshot]),
                        decision=decision,
                        direction=direction,
                    )
                )

        logs = self._load_notification_logs()
        logs.extend(messages)
        self._save_notification_logs(logs)
        return [{"message_id": item["message_id"], "recipient_contact": item["recipient_contact"]} for item in messages]

    def _build_notification_message(
        self,
        record_id: str,
        recipient_role: str,
        recipient_name: str,
        recipient_contact: str,
        related_item_ids: list[str],
        pushed_candidates: list[dict],
        decision: dict,
        direction: str,
    ) -> dict:
        return {
            "message_id": f"msg-{uuid.uuid4()}",
            "created_at": _now(),
            "status": "system_sent",
            "channel": "in_app",
            "direction": direction,
            "source_record_id": record_id,
            "recipient_role": recipient_role,
            "recipient_name": recipient_name,
            "recipient_contact": recipient_contact,
            "related_item_ids": related_item_ids,
            "pushed_candidates": pushed_candidates,
            "notification_type": decision.get("notification_type", "none"),
            "candidate_count": decision.get("candidate_count", len(decision.get("candidate_item_ids") or [])),
            "title": decision.get("message_title", "疑似找到匹配物品"),
            "body": decision.get("message_body", ""),
            "pickup_guide": decision.get("pickup_guide", ""),
            "reason": decision.get("reason", ""),
            "read_at": "",
            "handled_at": "",
            "user_feedback": "",
        }

    def _candidate_message_snapshot(self, candidates: list[dict]) -> list[dict]:
        rows = []
        for rank, item in enumerate(candidates, start=1):
            metadata = item.get("metadata", {})
            rows.append(
                {
                    "rank": item.get("rank", rank),
                    "item_id": item.get("item_id"),
                    "score": item.get("score"),
                    "distance": item.get("distance"),
                    "metadata": {
                        "record_type": metadata.get("record_type"),
                        "status": metadata.get("status"),
                        "category": metadata.get("category"),
                        "color": metadata.get("color"),
                        "location": metadata.get("location") or metadata.get("lost_location"),
                        "appearance": metadata.get("appearance"),
                        "image_path": metadata.get("image_path"),
                        "operator_name": metadata.get("operator_name"),
                        "operator_contact": metadata.get("operator_contact") or metadata.get("contact"),
                    },
                }
            )
        return rows

    def _load_agent_logs(self) -> list[dict]:
        if not self.agent_log_path.exists():
            return []
        try:
            data = json.loads(self.agent_log_path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def _save_agent_logs(self, logs: list[dict]) -> None:
        self.agent_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.agent_log_path.write_text(json.dumps(logs[-300:], ensure_ascii=False, indent=2), encoding="utf-8")

    def _record_agent_decision(
        self,
        record_id: str,
        direction: str,
        query_record: dict,
        candidates: list[dict],
        decision: dict,
    ) -> None:
        candidate_snapshot = []
        for item in candidates[:8]:
            metadata = item.get("metadata", {})
            candidate_snapshot.append(
                {
                    "item_id": item.get("item_id"),
                    "score": item.get("score"),
                    "category": metadata.get("category"),
                    "color": metadata.get("color"),
                    "location": metadata.get("location"),
                    "status": metadata.get("status"),
                    "appearance": metadata.get("appearance", ""),
                    "created_at": metadata.get("created_at"),
                }
            )
        log_entry = {
            "log_id": f"agent-{uuid.uuid4()}",
            "created_at": _now(),
            "record_id": record_id,
            "record_type": query_record.get("record_type", ""),
            "direction": direction,
            "agent_mode": self.matching_agent.mode,
            "prompt_version": "multi-candidate-close-score-v2",
            "query_summary": build_clip_search_string(query_record),
            "query_record": query_record,
            "candidate_count": len(candidates),
            "candidate_snapshot": candidate_snapshot,
            "decision": decision,
        }
        logs = self._load_agent_logs()
        logs.append(log_entry)
        self._save_agent_logs(logs)

    def _archive_image(self, image_path: str, item_id: str, kind: str) -> str:
        source = Path(image_path)
        if not source.exists():
            raise FileNotFoundError(f"图片文件不存在: {image_path}")
        suffix = source.suffix or ".jpg"
        target_dir = self.lost_upload_dir if kind == "lost" else self.upload_dir
        target = target_dir / f"{item_id}{suffix}"
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        return str(target)

    def _metadata_from_found_image(
        self,
        image_path: str,
        location: str,
        reporter_note: str,
        operator_name: str = "",
        operator_contact: str = "",
        image_sha256: str = "",
    ) -> dict:
        vlm_json = self.vision_extractor.extract_item_info(image_path, reporter_note, record_type="found")
        vlm_runtime = _vlm_runtime_metadata(vlm_json)
        if reporter_note:
            mark_field = "distinctive_marks" if "distinctive_marks" in vlm_json else "appearance"
            vlm_json[mark_field] = f"{vlm_json.get(mark_field, '')}；拾物补充：{reporter_note}"

        if vlm_json.get("is_identity_document"):
            identity = vlm_json.get("identity", {})
            metadata = {
                "record_type": "found",
                "location": location,
                "status": "待认领",
                "operator_name": operator_name or "未填写",
                "operator_contact": operator_contact or "未填写",
                "image_sha256": image_sha256,
                "is_identity_document": True,
                "category": vlm_json.get("category", "证件"),
                "color": "未知",
                "identity_student_id_partial": identity.get("student_id_partial", ""),
                "identity_name_masked": identity.get("name_masked", ""),
                "identity_school": identity.get("school", ""),
                "appearance": vlm_json.get("appearance", "证件类物品已脱敏。"),
                "distinctive_marks": reporter_note or "无",
                **vlm_runtime,
                "embedding_encoder": self.found_store.encoder.mode,
                "vlm_json": json.dumps(vlm_json, ensure_ascii=False),
            }
        else:
            metadata = {
                "record_type": "found",
                "location": location,
                "status": "待认领",
                "operator_name": operator_name or "未填写",
                "operator_contact": operator_contact or "未填写",
                "image_sha256": image_sha256,
                "is_identity_document": False,
                "category": vlm_json.get("category", "未知物品"),
                "subcategory": vlm_json.get("subcategory", "未知"),
                "color": vlm_json.get("color", "未知"),
                "material": vlm_json.get("material", "未知"),
                "size": vlm_json.get("size", "未知"),
                "condition": vlm_json.get("condition", "未知"),
                "brand": vlm_json.get("brand", "未知"),
                "logo": vlm_json.get("logo", "未识别"),
                "text_visible": vlm_json.get("text_visible", "无"),
                "distinctive_marks": vlm_json.get("distinctive_marks", reporter_note or "无"),
                "shape": vlm_json.get("shape", "未知"),
                "shape_profile": vlm_json.get("shape_profile", "未知"),
                "main_object": vlm_json.get("main_object", vlm_json.get("category", "未知物品")),
                "object_parts": vlm_json.get("object_parts", []),
                "interface_type": vlm_json.get("interface_type", "未知"),
                "pair_status": vlm_json.get("pair_status", "未知"),
                "accessories": vlm_json.get("accessories", "无"),
                "damage_or_wear": vlm_json.get("damage_or_wear", "未识别"),
                "location_hint": vlm_json.get("location_hint", "未知"),
                "search_keywords": vlm_json.get("search_keywords", []),
                "fine_grained_signature": vlm_json.get("fine_grained_signature", ""),
                "appearance": vlm_json.get("appearance", reporter_note or "暂无外观描述"),
                **vlm_runtime,
                "embedding_encoder": self.found_store.encoder.mode,
                "vlm_json": json.dumps(vlm_json, ensure_ascii=False),
            }
        metadata["created_at"] = _now()
        return metadata

    def _metadata_from_lost_report(
        self,
        description: str,
        lost_location: str = "",
        contact: str = "",
        query_image_path: Optional[str] = None,
        operator_name: str = "",
        operator_contact: str = "",
    ) -> dict:
        text = description or ""
        vlm_json = None
        if query_image_path:
            vlm_json = self.vision_extractor.extract_item_info(query_image_path, description, record_type="lost")
        vlm_runtime = _vlm_runtime_metadata(vlm_json)
        metadata = {
            "record_type": "lost",
            "status": "寻找中",
            "category": (vlm_json or {}).get("category") or _guess_category_from_text(text),
            "subcategory": (vlm_json or {}).get("subcategory", "未知"),
            "color": (vlm_json or {}).get("color") or _guess_color_from_text(text),
            "location": lost_location or "未知地点",
            "lost_location": lost_location or "未知地点",
            "contact": contact or "未填写",
            "operator_name": operator_name or "未填写",
            "operator_contact": operator_contact or contact or "未填写",
            "user_description": text,
            "material": (vlm_json or {}).get("material", "未知"),
            "size": (vlm_json or {}).get("size", "未知"),
            "condition": (vlm_json or {}).get("condition", "未知"),
            "brand": (vlm_json or {}).get("brand", "未知"),
            "logo": (vlm_json or {}).get("logo", "未识别"),
            "text_visible": (vlm_json or {}).get("text_visible", "无"),
            "shape": (vlm_json or {}).get("shape", "未知"),
            "shape_profile": (vlm_json or {}).get("shape_profile", "未知"),
            "main_object": (vlm_json or {}).get("main_object", (vlm_json or {}).get("category", "未知物品")),
            "object_parts": (vlm_json or {}).get("object_parts", []),
            "interface_type": (vlm_json or {}).get("interface_type", "未知"),
            "pair_status": (vlm_json or {}).get("pair_status", "未知"),
            "accessories": (vlm_json or {}).get("accessories", "无"),
            "damage_or_wear": (vlm_json or {}).get("damage_or_wear", "未识别"),
            "location_hint": (vlm_json or {}).get("location_hint", lost_location or "未知"),
            "search_keywords": (vlm_json or {}).get("search_keywords", []),
            "fine_grained_signature": (vlm_json or {}).get("fine_grained_signature", ""),
            "appearance": (vlm_json or {}).get("appearance") or text or "未填写描述",
            "distinctive_marks": (vlm_json or {}).get("distinctive_marks") or text or "无",
            **vlm_runtime,
            "embedding_encoder": self.lost_store.encoder.mode,
            "vlm_json": json.dumps(vlm_json or {"text_description": text}, ensure_ascii=False),
            "created_at": _now(),
        }
        if query_image_path:
            metadata["image_path"] = query_image_path
        return metadata

    def _add_to_store(
        self,
        store: VectorStoreManager,
        item_id: str,
        image_path: Optional[str],
        metadata: dict,
        image_weight: float,
        text_weight: float,
    ) -> None:
        search_text = build_clip_search_string(metadata)
        text_vector = store.encoder.encode_text(search_text)
        if image_path:
            image_vector = store.encoder.encode_image(image_path)
            fused_vector = fuse_vectors(image_vector, text_vector, image_weight=image_weight, text_weight=text_weight)
        else:
            image_vector = text_vector
            fused_vector = text_vector
        store.add_item(
            item_id=item_id,
            image_path=image_path or "",
            metadata=metadata,
            image_vector=image_vector,
            text_vector=text_vector,
            fused_vector=fused_vector,
        )

    def _format_candidates(self, search_results: dict) -> list[dict]:
        candidates = []
        for rank, (item_id, score, distance, metadata) in enumerate(zip(
            search_results["ids"],
            search_results["scores"],
            search_results["distances"],
            search_results["metadatas"],
        ), start=1):
            candidates.append(
                {
                    "rank": rank,
                    "item_id": item_id,
                    "score": score,
                    "distance": distance,
                    "metadata": metadata,
                }
            )
        return candidates

    def search_found_items(
        self,
        query_text: str = "",
        query_image_path: Optional[str] = None,
        location_filter: Optional[str] = None,
        top_k: int = 5,
        max_age_days: Optional[int] = 30,
    ) -> dict:
        results = self.found_store.search_item(
            query_text=query_text or None,
            query_image_path=query_image_path,
            top_k=top_k,
            location_filter=location_filter,
            status_filter="待认领",
            max_age_days=max_age_days,
        )
        return {**results, "candidates": self._format_candidates(results), "target_store": "found_items"}

    def search_lost_items(
        self,
        query_text: str = "",
        query_image_path: Optional[str] = None,
        top_k: int = 5,
        max_age_days: Optional[int] = 30,
    ) -> dict:
        results = self.lost_store.search_item(
            query_text=query_text or None,
            query_image_path=query_image_path,
            top_k=top_k,
            location_filter=None,
            status_filter="寻找中",
            max_age_days=max_age_days,
        )
        return {**results, "candidates": self._format_candidates(results), "target_store": "lost_items"}

    def report_found_item(
        self,
        image_path: str,
        location: str,
        reporter_note: str = "",
        operator_name: str = "",
        operator_contact: str = "",
        top_k: int = 5,
        use_agent: bool = False,
    ) -> dict:
        """捡到东西的人：入拾取物品库，并检索寻物品库。"""
        image_sha256 = _file_sha256(image_path)
        existing = self.found_store.find_by_metadata("image_sha256", image_sha256)
        if existing:
            raise ValueError(
                f"这张图片已经入过拾取物品库，已有编号：{existing['item_id']}。"
                "为避免重复记录，同一张图片不能重复上传。"
            )

        item_id = f"found-{uuid.uuid4()}"
        stored_image_path = self._archive_image(image_path, item_id, kind="found")
        metadata = self._metadata_from_found_image(
            stored_image_path,
            location,
            reporter_note,
            operator_name=operator_name,
            operator_contact=operator_contact,
            image_sha256=image_sha256,
        )
        self._add_to_store(
            self.found_store,
            item_id=item_id,
            image_path=stored_image_path,
            metadata=metadata,
            image_weight=0.6,
            text_weight=0.4,
        )
        metadata["image_path"] = stored_image_path

        query_text = build_clip_search_string(metadata)
        matches = self.search_lost_items(query_text=query_text, query_image_path=stored_image_path, top_k=top_k)
        agent_decision = self._evaluate_notification(
            query_record=metadata,
            candidates=matches["candidates"],
            enabled=use_agent,
            direction="found_to_lost",
            record_id=item_id,
        )
        return {
            "item_id": item_id,
            "record_type": "found",
            "features": metadata,
            "matches": matches["candidates"],
            "agent_decision": agent_decision,
            "fallback": matches.get("fallback"),
            "query_debug": {"source": "structured_metadata", "query_text": query_text},
            "vectors": {"store": "found_items", "fusion": "0.4 * image_vector + 0.6 * text_vector"},
        }

    def report_lost_item(
        self,
        description: str,
        lost_location: str = "",
        contact: str = "",
        query_image_path: Optional[str] = None,
        operator_name: str = "",
        operator_contact: str = "",
        top_k: int = 5,
        use_agent: bool = False,
    ) -> dict:
        """丢东西的人：入寻物品库，并检索拾取物品库。"""
        if not description and not query_image_path:
            raise ValueError("请至少填写物品描述或上传图片")

        item_id = f"lost-{uuid.uuid4()}"
        stored_image_path = self._archive_image(query_image_path, item_id, kind="lost") if query_image_path else None
        metadata = self._metadata_from_lost_report(
            description,
            lost_location,
            contact,
            stored_image_path,
            operator_name=operator_name,
            operator_contact=operator_contact,
        )
        self._add_to_store(
            self.lost_store,
            item_id=item_id,
            image_path=stored_image_path,
            metadata=metadata,
            image_weight=0.5,
            text_weight=0.5,
        )
        if stored_image_path:
            metadata["image_path"] = stored_image_path

        query_text = build_clip_search_string(metadata)
        matches = self.search_found_items(
            query_text=query_text,
            query_image_path=stored_image_path,
            location_filter=None,
            top_k=top_k,
        )
        agent_decision = self._evaluate_notification(
            query_record=metadata,
            candidates=matches["candidates"],
            enabled=use_agent,
            direction="lost_to_found",
            record_id=item_id,
        )
        return {
            "item_id": item_id,
            "record_type": "lost",
            "features": metadata,
            "matches": matches["candidates"],
            "agent_decision": agent_decision,
            "fallback": matches.get("fallback"),
            "query_debug": {"source": "structured_metadata", "query_text": query_text},
            "vectors": {"store": "lost_items", "fusion": "0.4 * image_vector + 0.6 * text_vector"},
        }

    def search_lost_item(
        self,
        query_text: str = "",
        expected_location: Optional[str] = None,
        query_image_path: Optional[str] = None,
        top_k: int = 5,
        max_age_days: Optional[int] = 30,
    ) -> dict:
        """兼容旧 UI：失主描述去拾取物品库检索。"""
        if not query_text and not query_image_path:
            raise ValueError("请至少输入文字描述或上传历史图片")
        results = self.search_found_items(
            query_text=query_text,
            query_image_path=query_image_path,
            location_filter=expected_location,
            top_k=top_k,
            max_age_days=max_age_days,
        )
        candidates = results["candidates"]
        return {
            "status": "success" if candidates else "not_found",
            "message": "" if candidates else "未找到相似拾取物",
            "candidates": candidates,
            "prefilter": results["prefilter"],
            "fallback": results.get("fallback"),
            "agent_decision": {
                "decision": "rag_only",
                "action": "manual_review",
                "pickup_guide": "当前只启用 RAG 召回，请人工核验 Top 候选。",
                "reason": "Agent 裁决后续接入",
            },
        }

    def confirm_pickup(self, item_id: str) -> bool:
        """核销拾取物。"""
        return self.found_store.update_item_metadata(
            item_id,
            {
                "status": "已认领",
                "resolved_at": _now(),
            },
        )

    def delete_found_item(self, item_id: str) -> bool:
        """管理员维护：删除拾取物品库记录。"""
        self.found_store.delete_item(item_id)
        return True

    def delete_lost_item(self, item_id: str) -> bool:
        """管理员维护：删除寻物品库记录。"""
        self.lost_store.delete_item(item_id)
        return True

    def clear_all_system_records(self, remove_upload_files: bool = False) -> dict:
        """管理员维护：清空业务记录，保留账号、地点配置、同义词和 API 配置。"""
        found_records = self.list_found_items()
        lost_records = self.list_lost_items()
        upload_paths = []
        if remove_upload_files:
            for record in [*found_records, *lost_records]:
                image_path = record.get("metadata", {}).get("image_path")
                if image_path:
                    upload_paths.append(Path(str(image_path)))

        found_count = self.found_store.clear_all_items()
        lost_count = self.lost_store.clear_all_items()
        cleared_logs = {}
        for name, path in [
            ("agent_decision_logs", self.agent_log_path),
            ("notification_messages", self.notification_log_path),
            ("resolution_logs", self.resolution_log_path),
        ]:
            if path.exists():
                try:
                    existing = json.loads(path.read_text(encoding="utf-8"))
                    cleared_logs[name] = len(existing) if isinstance(existing, list) else 1
                except (json.JSONDecodeError, OSError):
                    cleared_logs[name] = 0
            else:
                cleared_logs[name] = 0
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("[]", encoding="utf-8")

        removed_files = 0
        failed_files = 0
        if remove_upload_files:
            for image_path in dict.fromkeys(upload_paths):
                try:
                    if image_path.exists() and image_path.is_file():
                        image_path.unlink()
                        removed_files += 1
                except OSError:
                    failed_files += 1

        return {
            "found_records": found_count,
            "lost_records": lost_count,
            "logs": cleared_logs,
            "removed_upload_files": removed_files,
            "failed_upload_files": failed_files,
        }

    def inspect_vector_index_consistency(self) -> dict:
        """管理员维护：检查 JSON 业务记录和 Chroma 向量索引是否一致。"""
        return {
            "found": self.found_store.inspect_index_consistency(),
            "lost": self.lost_store.inspect_index_consistency(),
        }

    def rebuild_vector_indexes(self) -> dict:
        """管理员维护：以 JSON 业务记录为准重建 Chroma 向量索引。"""
        before = self.inspect_vector_index_consistency()
        found_result = self.found_store.rebuild_chroma_index()
        lost_result = self.lost_store.rebuild_chroma_index()
        after = self.inspect_vector_index_consistency()
        return {
            "before": before,
            "found": found_result,
            "lost": lost_result,
            "after": after,
        }

    def list_found_items(self, limit: Optional[int] = None) -> list[dict]:
        """查看拾取物品库中已有图片和记录。"""
        return self.found_store.list_items(limit=limit)

    def list_lost_items(self, limit: Optional[int] = None) -> list[dict]:
        """查看寻物品库中已有记录。"""
        return self.lost_store.list_items(limit=limit)

    def list_user_items(self, operator_name: str, limit: Optional[int] = None) -> list[dict]:
        """按账号查看用户历史上传记录，包含寻物和拾物。"""
        rows = []
        for record_type, records in [("found", self.list_found_items()), ("lost", self.list_lost_items())]:
            for record in records:
                metadata = record.get("metadata", {})
                if metadata.get("operator_name") == operator_name:
                    rows.append({**record, "record_type": record_type})
        rows.sort(key=lambda item: item.get("metadata", {}).get("created_at", ""), reverse=True)
        return rows[:limit] if limit else rows

    def list_agent_decision_logs(self, limit: Optional[int] = None) -> list[dict]:
        """管理员查看 DeepSeek 智能裁决审计记录。"""
        logs = self._load_agent_logs()
        logs.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return logs[:limit] if limit else logs

    def list_notification_logs(self, limit: Optional[int] = None) -> list[dict]:
        """管理员查看系统内通知发送记录。"""
        logs = self._load_notification_logs()
        logs.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return logs[:limit] if limit else logs

    def _notification_belongs_to_user(self, item: dict, account_id: str, phone: str = "") -> bool:
        """通知归属判断：优先严格按账号隔离，手机号只用于兼容没有账号的旧通知。"""
        account_id = str(account_id or "").strip()
        phone = str(phone or "").strip()
        recipient_name = str(item.get("recipient_name", "")).strip()
        recipient_contact = str(item.get("recipient_contact", "")).strip()
        if account_id:
            return recipient_name == account_id
        return bool(phone and not recipient_name and recipient_contact == phone)

    def list_user_notifications(
        self,
        account_id: str,
        phone: str = "",
        limit: Optional[int] = None,
    ) -> list[dict]:
        """普通用户查看自己账号收到的系统内通知。"""
        account_id = str(account_id or "").strip()
        phone = str(phone or "").strip()
        logs = []
        for item in self._load_notification_logs():
            if self._notification_belongs_to_user(item, account_id, phone):
                logs.append(item)
        logs.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return logs[:limit] if limit else logs

    def mark_notification_read(self, message_id: str) -> bool:
        logs = self._load_notification_logs()
        changed = False
        for item in logs:
            if item.get("message_id") == message_id and not item.get("read_at"):
                item["read_at"] = _now()
                changed = True
        if changed:
            self._save_notification_logs(logs)
        return changed

    def _manual_claim_message(self, found_item_id: str, found_record: dict, account_id: str, phone: str, status: str = "system_sent") -> dict:
        metadata = found_record.get("metadata", {})
        item_name = " · ".join(
            part
            for part in [
                str(metadata.get("category", "")).strip() or "未知物品",
                str(metadata.get("color", "")).strip() or "未知颜色",
                str(metadata.get("location", "")).strip() or "未知地点",
            ]
            if part
        )
        now = _now()
        return {
            "message_id": f"msg-{uuid.uuid4()}",
            "created_at": now,
            "status": status,
            "channel": "in_app",
            "direction": "manual_claim",
            "source_record_id": found_item_id,
            "recipient_role": "lost_owner",
            "recipient_name": account_id,
            "recipient_contact": phone or "未填写",
            "related_item_ids": [],
            "pushed_candidates": [
                {
                    "rank": 1,
                    "item_id": found_item_id,
                    "score": 1.0,
                    "distance": 0.0,
                    "metadata": {
                        "record_type": metadata.get("record_type"),
                        "status": metadata.get("status"),
                        "category": metadata.get("category"),
                        "color": metadata.get("color"),
                        "location": metadata.get("location"),
                        "appearance": metadata.get("appearance"),
                        "image_path": metadata.get("image_path"),
                        "operator_name": metadata.get("operator_name"),
                        "operator_contact": metadata.get("operator_contact") or metadata.get("contact"),
                    },
                }
            ],
            "notification_type": "manual_claim",
            "candidate_count": 1,
            "title": "待办：确认招领物品",
            "body": f"你标记了“{item_name}”似乎是你的。请线下核验并在取回后点击“已取回”。",
            "pickup_guide": "请核验图片、接口、磨损、文字标识等细节，确认取回后系统会把该物品标记为已认领。",
            "reason": "用户在招领区主动标记似乎是我的。",
            "read_at": now if status in {"confirmed", "rejected"} else "",
            "handled_at": now if status in {"confirmed", "rejected"} else "",
            "user_feedback": "picked_up" if status == "confirmed" else "",
        }

    def create_manual_claim_notification(self, found_item_id: str, account_id: str, phone: str = "") -> dict:
        """用户在招领区主动把某条记录加入自己的待办通知。"""
        account_id = str(account_id or "").strip()
        phone = str(phone or "").strip()
        if not account_id:
            raise ValueError("请先登录后再标记物品")
        found_record = self.found_store.get_item(found_item_id)
        if not found_record:
            raise ValueError("招领记录不存在")
        metadata = found_record.get("metadata", {})
        if metadata.get("status") == "已认领":
            raise ValueError("该物品已被认领")

        logs = self._load_notification_logs()
        for item in logs:
            same_owner = self._notification_belongs_to_user(item, account_id, phone)
            if (
                item.get("notification_type") == "manual_claim"
                and item.get("source_record_id") == found_item_id
                and item.get("status") == "system_sent"
                and same_owner
            ):
                return item

        message = self._manual_claim_message(found_item_id, found_record, account_id, phone)
        logs.append(message)
        self._save_notification_logs(logs)
        return message

    def cancel_manual_claim_notification(self, found_item_id: str, account_id: str, phone: str = "") -> int:
        """取消用户主动标记产生的未处理待办通知。"""
        account_id = str(account_id or "").strip()
        phone = str(phone or "").strip()
        logs = self._load_notification_logs()
        kept = []
        removed_count = 0
        for item in logs:
            same_owner = self._notification_belongs_to_user(item, account_id, phone)
            is_pending_manual_claim = (
                item.get("notification_type") == "manual_claim"
                and item.get("source_record_id") == found_item_id
                and item.get("status") == "system_sent"
                and same_owner
            )
            if is_pending_manual_claim:
                removed_count += 1
                continue
            kept.append(item)
        if removed_count:
            self._save_notification_logs(kept)
        return removed_count

    def confirm_manual_claim(
        self,
        found_item_id: str,
        account_id: str,
        phone: str = "",
        message_id: str = "",
        lost_item_id: str = "",
    ) -> dict:
        """用户确认已经取回招领物，更新通知、物品状态和闭环审计。"""
        account_id = str(account_id or "").strip()
        phone = str(phone or "").strip()
        lost_item_id = str(lost_item_id or "").strip()
        if not account_id:
            raise ValueError("请先登录后再确认取回")
        found_record = self.found_store.get_item(found_item_id)
        if not found_record:
            raise ValueError("招领记录不存在")
        if found_record.get("metadata", {}).get("status") == "已认领":
            raise ValueError("该物品已被认领")
        lost_record = self.lost_store.get_item(lost_item_id) if lost_item_id else None

        now = _now()
        resolution_id = f"resolution-{uuid.uuid4()}"
        self.found_store.update_item_metadata(
            found_item_id,
            {
                "status": "已认领",
                "claimed_by_account": account_id,
                "claimed_by_phone": phone,
                "claimed_at": now,
                "resolved_at": now,
                "resolution_id": resolution_id,
                "matched_lost_item_id": lost_item_id,
            },
        )
        if lost_item_id:
            self.lost_store.update_item_metadata(
                lost_item_id,
                {
                    "status": "已找回",
                    "resolved_at": now,
                    "resolution_id": resolution_id,
                    "matched_found_item_id": found_item_id,
                },
            )

        logs = self._load_notification_logs()
        handled_messages = []
        for item in logs:
            same_owner = self._notification_belongs_to_user(item, account_id, phone)
            is_target = bool(message_id and item.get("message_id") == message_id)
            is_manual_same_item = (
                item.get("notification_type") == "manual_claim"
                and item.get("source_record_id") == found_item_id
                and same_owner
            )
            candidate_ids = {str(candidate.get("item_id", "")) for candidate in item.get("pushed_candidates", [])}
            is_auto_same_pair = (
                lost_item_id
                and same_owner
                and item.get("source_record_id") == lost_item_id
                and (found_item_id in set(item.get("related_item_ids", [])) or found_item_id in candidate_ids)
            )
            if is_target or is_manual_same_item or is_auto_same_pair:
                item["status"] = "confirmed"
                item["user_feedback"] = "picked_up"
                item["handled_at"] = now
                item["read_at"] = item.get("read_at") or now
                item["selected_item_id"] = found_item_id
                item["selected_rank"] = next(
                    (
                        candidate.get("rank")
                        for candidate in item.get("pushed_candidates", [])
                        if candidate.get("item_id") == found_item_id
                    ),
                    1,
                )
                handled_messages.append(item)

        if not handled_messages:
            message = self._manual_claim_message(found_item_id, found_record, account_id, phone, status="confirmed")
            message["selected_item_id"] = found_item_id
            message["selected_rank"] = 1
            logs.append(message)
            handled_messages.append(message)

        self._save_notification_logs(logs)
        user_record = self.get_item_record(lost_item_id) if lost_item_id else None
        resolution = {
            "resolution_id": resolution_id,
            "created_at": now,
            "message_id": handled_messages[0].get("message_id", ""),
            "result": "manual_claim_picked_up",
            "selected_item_id": found_item_id,
            "selected_rank": handled_messages[0].get("selected_rank", 1),
            "selected_candidate_rank": handled_messages[0].get("selected_rank", 1),
            "source_record_id": lost_item_id or found_item_id,
            "related_item_ids": [found_item_id] if lost_item_id else [],
            "notification": handled_messages[0],
            "user_records": [user_record or lost_record] if (user_record or lost_record) else [],
            "pushed_records": [self.get_item_record(found_item_id) or found_record],
            "agent_decision_log": {},
        }
        resolution_logs = self._load_resolution_logs()
        resolution_logs.append(resolution)
        self._save_resolution_logs(resolution_logs)
        return resolution

    def delete_user_notification(self, message_id: str, account_id: str, phone: str = "") -> bool:
        """普通用户只能删除属于自己且已处理的通知，管理员审计日志不受影响。"""
        account_id = str(account_id or "").strip()
        phone = str(phone or "").strip()
        logs = self._load_notification_logs()
        kept = []
        deleted = False
        for item in logs:
            is_target = item.get("message_id") == message_id
            is_owner = self._notification_belongs_to_user(item, account_id, phone)
            is_handled = item.get("status") in {"confirmed", "rejected"}
            if is_target and is_owner and is_handled:
                deleted = True
                continue
            kept.append(item)
        if deleted:
            self._save_notification_logs(kept)
        return deleted

    def delete_user_notifications(self, message_ids: list[str], account_id: str, phone: str = "") -> int:
        deleted_count = 0
        for message_id in message_ids:
            if self.delete_user_notification(message_id, account_id, phone):
                deleted_count += 1
        return deleted_count

    def get_item_record(self, item_id: str) -> Optional[dict]:
        if item_id.startswith("found-"):
            record = self.found_store.get_item(item_id)
            return {**record, "record_type": "found"} if record else None
        if item_id.startswith("lost-"):
            record = self.lost_store.get_item(item_id)
            return {**record, "record_type": "lost"} if record else None
        return None

    def get_notification_detail(self, message_id: str) -> Optional[dict]:
        message = next((item for item in self._load_notification_logs() if item.get("message_id") == message_id), None)
        if not message:
            return None
        source_record = self.get_item_record(message.get("source_record_id", ""))
        related_records = [
            record
            for item_id in message.get("related_item_ids", [])
            for record in [self.get_item_record(item_id)]
            if record
        ]
        if message.get("notification_type") == "manual_claim" or message.get("direction") == "manual_claim":
            return {
                "message": message,
                "user_records": [],
                "pushed_records": [source_record] if source_record else [],
            }
        if message.get("direction") == "lost_to_found":
            user_records = [source_record] if source_record else []
            pushed_records = related_records
        else:
            user_records = related_records
            pushed_records = [source_record] if source_record else []
        return {
            "message": message,
            "user_records": user_records,
            "pushed_records": pushed_records,
        }

    def handle_notification_feedback(self, message_id: str, selected_item_id: str = "", matched: bool = False) -> dict:
        logs = self._load_notification_logs()
        message = next((item for item in logs if item.get("message_id") == message_id), None)
        if not message:
            raise ValueError("通知不存在")
        detail = self.get_notification_detail(message_id) or {}
        selected_rank = None
        for candidate in message.get("pushed_candidates", []):
            if candidate.get("item_id") == selected_item_id:
                selected_rank = candidate.get("rank")
                break
        if selected_rank is None and matched and message.get("pushed_candidates"):
            selected_rank = message["pushed_candidates"][0].get("rank")
        resolution = {
            "resolution_id": f"resolution-{uuid.uuid4()}",
            "created_at": _now(),
            "message_id": message_id,
            "result": "matched" if matched else "badcase",
            "selected_item_id": selected_item_id,
            "selected_rank": selected_rank,
            "selected_candidate_rank": selected_rank,
            "source_record_id": message.get("source_record_id"),
            "related_item_ids": message.get("related_item_ids", []),
            "notification": message,
            "user_records": detail.get("user_records", []),
            "pushed_records": detail.get("pushed_records", []),
            "agent_decision_log": self._find_agent_log(message.get("source_record_id", "")),
        }
        if matched and selected_item_id:
            self._resolve_pair(message, selected_item_id, resolution["resolution_id"])
            message["status"] = "confirmed"
            message["user_feedback"] = "confirmed_mine"
        else:
            message["status"] = "rejected"
            message["user_feedback"] = "not_mine"
        message["handled_at"] = _now()
        if not message.get("read_at"):
            message["read_at"] = message["handled_at"]
        self._save_notification_logs(logs)
        resolution_logs = self._load_resolution_logs()
        resolution_logs.append(resolution)
        self._save_resolution_logs(resolution_logs)
        return resolution

    def _find_agent_log(self, record_id: str) -> dict:
        for log in self._load_agent_logs():
            if log.get("record_id") == record_id:
                return log
        return {}

    def _resolve_pair(self, message: dict, selected_item_id: str, resolution_id: str) -> None:
        source_id = message.get("source_record_id", "")
        if message.get("direction") == "lost_to_found":
            lost_id, found_id = source_id, selected_item_id
        else:
            found_id = selected_item_id or source_id
            lost_id = (message.get("related_item_ids") or [""])[0]
        resolved_at = _now()
        found_updates = {"status": "已认领", "resolved_at": resolved_at, "resolution_id": resolution_id, "matched_lost_item_id": lost_id}
        lost_updates = {"status": "已找回", "resolved_at": resolved_at, "resolution_id": resolution_id, "matched_found_item_id": found_id}
        if found_id:
            self.found_store.update_item_metadata(found_id, found_updates)
        if lost_id:
            self.lost_store.update_item_metadata(lost_id, lost_updates)

    def list_resolution_logs(self, limit: Optional[int] = None) -> list[dict]:
        logs = self._load_resolution_logs()
        logs.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return logs[:limit] if limit else logs

    def rematch_user_item(self, item_id: str, record_type: str, top_k: int = 5) -> list[dict]:
        """点击历史记录时，重新检索另一侧库并返回最新匹配排序。"""
        if record_type == "found":
            record = self.found_store.get_item(item_id)
            if not record:
                raise ValueError("拾取记录不存在或已被删除")
            query_vector = record.get("vectors", {}).get("fused")
            if not query_vector:
                raise ValueError("拾取记录缺少入库向量，请重新上传或重建索引")
            results = self.lost_store.search_vector(
                query_vector=query_vector,
                top_k=top_k,
                location_filter=None,
                status_filter="寻找中",
                max_age_days=30,
            )
            return self._format_candidates({**results, "target_store": "lost_items"})

        if record_type == "lost":
            record = self.lost_store.get_item(item_id)
            if not record:
                raise ValueError("寻物记录不存在或已被删除")
            query_vector = record.get("vectors", {}).get("fused")
            if not query_vector:
                raise ValueError("寻物记录缺少入库向量，请重新上传或重建索引")
            results = self.found_store.search_vector(
                query_vector=query_vector,
                top_k=top_k,
                location_filter=None,
                status_filter="待认领",
                max_age_days=30,
            )
            return self._format_candidates({**results, "target_store": "found_items"})

        raise ValueError(f"未知记录类型：{record_type}")

    def get_system_status(self) -> dict:
        consistency = self.inspect_vector_index_consistency()
        return {
            "found_count": self.found_store.get_collection_count(),
            "lost_count": self.lost_store.get_collection_count(),
            "items_count": self.found_store.get_collection_count(),
            "vector_backend": self.found_store.backend,
            "clip_mode": self.found_store.encoder.mode,
            "vision_mode": self.vision_extractor.mode,
            "agent_mode": self.matching_agent.mode,
            "data_dir": str(self.data_dir),
            "found_store_dir": str(self.found_store.persist_directory),
            "lost_store_dir": str(self.lost_store.persist_directory),
            "index_consistency": {
                "found_missing_in_chroma": len(consistency["found"].get("missing_in_chroma", [])),
                "found_extra_in_chroma": len(consistency["found"].get("extra_in_chroma", [])),
                "lost_missing_in_chroma": len(consistency["lost"].get("missing_in_chroma", [])),
                "lost_extra_in_chroma": len(consistency["lost"].get("extra_in_chroma", [])),
            },
        }

    def seed_demo_data(self) -> list[dict]:
        """生成演示拾取物和寻物登记，便于端到端验证双库召回。"""
        from PIL import Image, ImageDraw

        demo_dir = self.data_dir / "demo_images"
        demo_dir.mkdir(parents=True, exist_ok=True)
        found_samples = [
            ("blue_cup.png", "二食堂前台", "蓝色水杯，杯身有熊猫贴纸", (45, 105, 210)),
            ("white_earphone.png", "图书馆失物招领处", "白色无线耳机，带充电盒", (235, 235, 230)),
            ("black_key.png", "东门保卫室", "黑色钥匙串，挂着圆形吊牌", (30, 30, 35)),
        ]
        results = []
        for filename, location, note, color in found_samples:
            path = demo_dir / filename
            if not path.exists():
                img = Image.new("RGB", (480, 320), (248, 250, 255))
                draw = ImageDraw.Draw(img)
                draw.rounded_rectangle((140, 70, 340, 250), radius=28, fill=color)
                draw.text((120, 270), filename.replace("_", " "), fill=(20, 30, 60))
                img.save(path)
            results.append(self.report_found_item(str(path), location, note))

        results.append(
            self.report_lost_item(
                description="我在食堂丢了一个蓝色水杯，上面有熊猫贴纸",
                lost_location="二食堂附近",
                contact="demo_user",
            )
        )
        return results


if __name__ == "__main__":
    system = CampusSystemCore()
    if system.get_system_status()["found_count"] == 0:
        system.seed_demo_data()
    print(system.search_lost_item("我在食堂丢了一个蓝色水杯，上面有熊猫贴纸"))
