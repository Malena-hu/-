"""
vector_db.py
失物招领智能检索系统 - 中文图文编码、ChromaDB 存储与混合检索层
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

VECTOR_DIM = 512
DEFAULT_CHINESE_CLIP_MODEL = "OFA-Sys/chinese-clip-vit-base-patch16"
DEFAULT_SIGLIP_MODEL = "google/siglip-base-patch16-224"
DEFAULT_CLIP_MODEL = os.getenv("VISION_EMBEDDING_MODEL", DEFAULT_CHINESE_CLIP_MODEL)
DEFAULT_ENCODER_BACKEND = os.getenv("VISION_ENCODER_BACKEND", "auto").lower()


class NoOpEmbeddingFunction:
    def __call__(self, texts):
        return [[0.0] * VECTOR_DIM for _ in texts]


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector if norm == 0 else vector / norm


def _fit_vector_dim(vector: np.ndarray, dim: int = VECTOR_DIM) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    if vector.size == dim:
        return vector
    if vector.size > dim:
        return vector[:dim]
    padded = np.zeros(dim, dtype=np.float32)
    padded[: vector.size] = vector
    return padded


def _stable_token_vector(token: str, dim: int = VECTOR_DIM) -> np.ndarray:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big", signed=False)
    return np.random.default_rng(seed).normal(0, 1, dim)


def encode_text_locally(text: str, dim: int = VECTOR_DIM) -> list[float]:
    if not text or not text.strip():
        raise ValueError("输入文本不能为空")
    compact = "".join(ch for ch in text.lower() if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")
    tokens = [compact[i : i + 2] for i in range(max(1, len(compact) - 1))]
    vector = np.zeros(dim, dtype=np.float32)
    for token in tokens:
        vector += _stable_token_vector(token, dim)
    return _l2_normalize(vector).astype(float).tolist()


def encode_image_locally(image_path: str, dim: int = VECTOR_DIM) -> list[float]:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")
    image = Image.open(image_path).convert("RGB").resize((64, 64))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    stats = [arr.mean(axis=(0, 1)), arr.std(axis=(0, 1))]
    for channel in range(3):
        stats.append(np.histogram(arr[:, :, channel], bins=16, range=(0, 1), density=True)[0])
    small = np.concatenate(stats)
    vector = np.zeros(dim, dtype=np.float32)
    vector[: len(small)] = small
    vector += 0.05 * _stable_token_vector(hashlib.sha256(Path(image_path).read_bytes()).hexdigest(), dim)
    return _l2_normalize(vector).astype(float).tolist()


def fuse_vectors(
    image_vector: list[float],
    text_vector: list[float],
    image_weight: float = 0.6,
    text_weight: float = 0.4,
) -> list[float]:
    fused = image_weight * np.asarray(image_vector) + text_weight * np.asarray(text_vector)
    return _l2_normalize(fused).astype(float).tolist()


def cosine_similarity(left: list[float], right: list[float]) -> float:
    a = np.asarray(left, dtype=np.float32)
    b = np.asarray(right, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denom == 0 else float(np.dot(a, b) / denom)


def build_clip_search_string(metadata: dict) -> str:
    """为 CLIP 召回层构建紧凑文本，完整 JSON 留给 DeepSeek 精排。"""
    priorities = [
        ("subcategory", "细类"),
        ("category", ""),
        ("main_object", "主体"),
        ("fine_grained_signature", "细节"),
        ("object_parts", "部件"),
        ("shape_profile", "轮廓"),
        ("interface_type", "接口"),
        ("pair_status", "成套"),
        ("color", ""),
        ("shape", "形状"),
        ("brand", "品牌"),
        ("logo", "图案"),
        ("distinctive_marks", "标记"),
        ("text_visible", "文字"),
        ("accessories", "附件"),
        ("damage_or_wear", "磨损"),
        ("search_keywords", "关键词"),
        ("appearance", ""),
    ]
    parts: list[str] = []
    for key, label in priorities:
        raw_value = metadata.get(key, "")
        value = "，".join(raw_value) if isinstance(raw_value, list) else str(raw_value).strip()
        if not value or value in {"未知", "未识别", "无"}:
            continue
        segment = f"{label}{value}" if label else value
        parts.append(segment)
        if len("，".join(parts)) >= 60:
            break
    return "，".join(parts)[:180] or "未知物品"


def _metadata_text(metadata: dict) -> str:
    values = []
    for key in [
        "subcategory",
        "category",
        "main_object",
        "object_parts",
        "interface_type",
        "pair_status",
        "shape_profile",
        "shape",
        "accessories",
        "search_keywords",
        "fine_grained_signature",
        "appearance",
        "distinctive_marks",
        "clip_search_text",
    ]:
        value = metadata.get(key, "")
        if isinstance(value, list):
            values.extend(str(item) for item in value)
        else:
            values.append(str(value))
    return " ".join(values).lower()


def infer_fine_subcategory(metadata: dict) -> str:
    explicit = str(metadata.get("subcategory", "")).strip()
    if explicit and explicit not in {"未知", "未知物品", "其他物品"}:
        return explicit
    text = _metadata_text(metadata)
    rules = [
        ("耳机盒", ["耳机盒", "充电盒", "耳机仓"]),
        ("无线耳机", ["无线耳机", "蓝牙耳机", "耳塞", "耳柄", "入耳式耳机", "半入耳式"]),
        ("有线耳机", ["有线耳机", "耳机线"]),
        ("充电线", ["充电线", "数据线", "线缆", "usb-c线", "type-c线", "lightning线", "插头线"]),
        ("充电器", ["充电器", "充电头", "适配器", "电源适配器"]),
        ("手机", ["手机"]),
        ("手表", ["手表", "电子表", "表带", "表盘"]),
        ("水杯", ["水杯", "保温杯", "杯盖", "杯身"]),
        ("钥匙", ["钥匙", "钥匙串", "钥匙圈"]),
    ]
    for category, keywords in rules:
        if any(keyword in text for keyword in keywords):
            return category
    return str(metadata.get("category", "未知物品")).strip() or "未知物品"


def _metadata_overlap(left: dict, right: dict, key: str) -> bool:
    def to_set(value) -> set[str]:
        if isinstance(value, list):
            return {str(item).strip().lower() for item in value if str(item).strip()}
        return {part for part in str(value).lower().replace("，", " ").replace("、", " ").split() if part}

    left_set = to_set(left.get(key, ""))
    right_set = to_set(right.get(key, ""))
    return bool(left_set and right_set and left_set.intersection(right_set))


def metadata_rerank_score(base_score: float, query_metadata: Optional[dict], candidate_metadata: dict) -> float:
    if not query_metadata:
        return base_score
    adjusted = float(base_score)
    query_sub = infer_fine_subcategory(query_metadata)
    candidate_sub = infer_fine_subcategory(candidate_metadata)
    unknowns = {"", "未知", "未知物品", "其他物品"}
    if query_sub not in unknowns and candidate_sub not in unknowns:
        if query_sub == candidate_sub:
            adjusted += 0.08
        elif query_sub in {"无线耳机", "有线耳机", "耳机盒"} or candidate_sub in {"无线耳机", "有线耳机", "耳机盒"}:
            adjusted -= 0.22
        elif query_sub in {"充电线", "充电器"} or candidate_sub in {"充电线", "充电器"}:
            adjusted -= 0.16
    if _metadata_overlap(query_metadata, candidate_metadata, "object_parts"):
        adjusted += 0.04
    query_interface = str(query_metadata.get("interface_type", "")).strip()
    candidate_interface = str(candidate_metadata.get("interface_type", "")).strip()
    if query_interface and candidate_interface and query_interface not in {"未知", "无"} and candidate_interface not in {"未知", "无"}:
        adjusted += 0.04 if query_interface == candidate_interface else -0.05
    return max(-1.0, min(1.0, adjusted))


class ClipEncoder:
    """中文图文编码器优先，失败时退回本地稳定向量。"""

    def __init__(self, model_name: str = DEFAULT_CLIP_MODEL, backend: str = DEFAULT_ENCODER_BACKEND):
        self.model_name = model_name
        self.backend = backend
        self.model = None
        self.processor = None
        self.torch = None
        self.model_backend = "local"
        self.mode = "local-hash-fallback"
        if os.getenv("DISABLE_SIGLIP", os.getenv("DISABLE_REAL_CLIP", "0")) == "1":
            return
        self._load_model()

    def _allow_model_download(self) -> bool:
        return os.getenv("ALLOW_MODEL_DOWNLOAD", os.getenv("ALLOW_SIGLIP_DOWNLOAD", "0")) == "1"

    def _load_model(self) -> None:
        candidates: list[tuple[str, str]] = []
        if self.backend in {"auto", "chinese_clip", "chinese-clip"}:
            candidates.append(("chinese_clip", self.model_name or DEFAULT_CHINESE_CLIP_MODEL))
        if self.backend in {"auto", "siglip"}:
            siglip_name = (
                self.model_name
                if self.backend == "siglip" and self.model_name != DEFAULT_CHINESE_CLIP_MODEL
                else DEFAULT_SIGLIP_MODEL
            )
            candidates.append(("siglip", siglip_name))

        errors: list[str] = []
        for backend, model_name in candidates:
            try:
                self._load_backend(backend, model_name)
                return
            except Exception as exc:
                errors.append(f"{backend}:{model_name} -> {exc}")

        if errors:
            print("[VisionEncoder] 模型加载失败，使用本地兜底：" + " | ".join(errors), flush=True)

    def _load_backend(self, backend: str, model_name: str) -> None:
        import torch

        local_files_only = not self._allow_model_download()
        if backend == "chinese_clip":
            from transformers import (
                AutoTokenizer,
                ChineseCLIPImageProcessor,
                ChineseCLIPModel,
                ChineseCLIPProcessor,
            )

            try:
                processor = ChineseCLIPProcessor.from_pretrained(model_name, local_files_only=local_files_only)
            except Exception:
                try:
                    image_processor = ChineseCLIPImageProcessor.from_pretrained(
                        model_name,
                        local_files_only=local_files_only,
                    )
                except Exception:
                    image_processor = ChineseCLIPImageProcessor(
                        size={"shortest_edge": 224},
                        crop_size={"height": 224, "width": 224},
                        image_mean=[0.48145466, 0.4578275, 0.40821073],
                        image_std=[0.26862954, 0.26130258, 0.27577711],
                    )
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    local_files_only=local_files_only,
                )
                processor = ChineseCLIPProcessor(image_processor=image_processor, tokenizer=tokenizer)
            model = ChineseCLIPModel.from_pretrained(
                model_name,
                local_files_only=local_files_only,
                use_safetensors=False,
            )
        elif backend == "siglip":
            from transformers import SiglipModel, SiglipProcessor

            processor = SiglipProcessor.from_pretrained(model_name, local_files_only=local_files_only)
            model = SiglipModel.from_pretrained(model_name, local_files_only=local_files_only)
        else:
            raise ValueError(f"不支持的图文向量后端: {backend}")

        model.eval()
        self.torch = torch
        self.processor = processor
        self.model = model
        self.model_name = model_name
        self.model_backend = backend
        self.mode = f"{backend}:{model_name}"
        print(f"[VisionEncoder] 已加载 {self.mode}", flush=True)

    def encode_text(self, text: str) -> list[float]:
        if self.model is None:
            return encode_text_locally(text)
        try:
            padding = "max_length" if self.model_backend == "siglip" else True
            inputs = self.processor(text=[text], padding=padding, return_tensors="pt")
            with self.torch.no_grad():
                if self.model_backend == "chinese_clip":
                    features = self.model.get_text_features(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask"),
                        token_type_ids=inputs.get("token_type_ids"),
                    )
                else:
                    features = self.model.get_text_features(**inputs)
            vector = _fit_vector_dim(features[0].detach().cpu().numpy())
            return _l2_normalize(vector).astype(float).tolist()
        except Exception as exc:
            print(f"[VisionEncoder] 文本向量化失败，使用本地兜底：{exc}", flush=True)
            return encode_text_locally(text)

    def encode_image(self, image_path: str) -> list[float]:
        if self.model is None:
            return encode_image_locally(image_path)
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            with self.torch.no_grad():
                if self.model_backend == "chinese_clip":
                    features = self.model.get_image_features(pixel_values=inputs["pixel_values"])
                else:
                    features = self.model.get_image_features(**inputs)
            vector = _fit_vector_dim(features[0].detach().cpu().numpy())
            return _l2_normalize(vector).astype(float).tolist()
        except Exception as exc:
            print(f"[VisionEncoder] 图片向量化失败，使用本地兜底：{exc}", flush=True)
            return encode_image_locally(image_path)


class VectorStoreManager:
    """ChromaDB 持久化向量库，保留 JSON 后备实现。"""

    def __init__(
        self,
        persist_directory: str = "./runtime_data/chroma_db",
        collection_name: str = "campus_lost_found_fused",
        model_name: str = DEFAULT_CLIP_MODEL,
        encoder: Optional[ClipEncoder] = None,
    ):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.encoder = encoder or ClipEncoder(model_name)
        self.backend = "json"
        self.client = None
        self.collection = None
        self.json_path = self.persist_directory / "items.json"
        self.items = self._load_json_items()
        self._init_chromadb()
        print(f"[VectorStore] backend={self.backend}, encoder={self.encoder.mode}, count={self.get_collection_count()}")

    def _open_chroma_collection(self):
        if self.client is None:
            return None
        return self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=NoOpEmbeddingFunction(),
            metadata={"hnsw:space": "cosine"},
        )

    def _init_chromadb(self) -> None:
        if os.getenv("DISABLE_CHROMADB", "0") == "1":
            return
        try:
            import chromadb

            if hasattr(chromadb, "PersistentClient"):
                client = chromadb.PersistentClient(path=str(self.persist_directory))
            else:
                from chromadb.config import Settings

                client = chromadb.Client(
                    Settings(
                        chroma_db_impl="duckdb+parquet",
                        persist_directory=str(self.persist_directory),
                    )
                )
            self.client = client
            self.collection = self._open_chroma_collection()
            self.backend = "chromadb"
        except Exception as exc:
            print(f"[ChromaDB] 初始化失败，使用 JSON 后备：{exc}")

    def _load_json_items(self) -> dict[str, dict]:
        if not self.json_path.exists():
            return {}
        try:
            return json.loads(self.json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _save_json_items(self) -> None:
        self.json_path.write_text(json.dumps(self.items, ensure_ascii=False, indent=2), encoding="utf-8")

    def _metadata_for_chroma(self, metadata: dict) -> dict:
        clean = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                clean[key] = "" if value is None else value
            else:
                clean[key] = json.dumps(value, ensure_ascii=False)
        return clean

    def _chroma_ids(self) -> list[str]:
        if self.collection is None or not hasattr(self.collection, "get"):
            return []
        try:
            return list(self.collection.get().get("ids", []))
        except Exception as exc:
            print(f"[ChromaDB] 读取集合 ID 失败：{exc}", flush=True)
            return []

    def _add_record_to_chroma(self, item_id: str, record: dict) -> None:
        if self.collection is None:
            raise RuntimeError("Chroma collection is not initialized")
        metadata = record.get("metadata", {})
        fused_vector = record.get("vectors", {}).get("fused")
        if not fused_vector:
            raise ValueError("缺少 fused 向量")
        document = metadata.get("clip_search_text") or build_clip_search_string(metadata)
        self.collection.add(
            ids=[item_id],
            embeddings=[fused_vector],
            metadatas=[self._metadata_for_chroma(metadata)],
            documents=[document],
        )

    def add_item(
        self,
        item_id: str,
        image_path: str,
        metadata: dict,
        image_vector: Optional[list[float]] = None,
        text_vector: Optional[list[float]] = None,
        fused_vector: Optional[list[float]] = None,
    ) -> None:
        image_vector = image_vector or self.encoder.encode_image(image_path)
        text_vector = text_vector or self.encoder.encode_text(build_clip_search_string(metadata))
        fused_vector = fused_vector or fuse_vectors(image_vector, text_vector)
        now = datetime.now().isoformat(timespec="seconds")
        stored_metadata = {
            **metadata,
            "image_path": image_path,
            "clip_search_text": build_clip_search_string(metadata),
            "created_at": metadata.get("created_at", now),
            "updated_at": now,
        }

        self.items[item_id] = {
            "item_id": item_id,
            "metadata": stored_metadata,
            "vectors": {"image": image_vector, "text": text_vector, "fused": fused_vector},
        }
        self._save_json_items()

        if self.backend == "chromadb" and self.collection is not None:
            try:
                self.collection.delete(ids=[item_id])
            except Exception:
                pass
            document = stored_metadata.get("clip_search_text", "")
            try:
                self.collection.add(
                    ids=[item_id],
                    embeddings=[fused_vector],
                    metadatas=[self._metadata_for_chroma(stored_metadata)],
                    documents=[document],
                )
                if self.client is not None and hasattr(self.client, "persist"):
                    self.client.persist()
            except Exception as exc:
                print(f"[ChromaDB] 写入失败，仅保留 JSON 后备：{exc}")
        print(f"[VectorStore] 写入 item_id={item_id}")

    def rebuild_chroma_index(self, backup_json: bool = True) -> dict:
        """Rebuild Chroma from the JSON mirror, keeping JSON as the source of truth."""
        result = {
            "collection": self.collection_name,
            "backend": self.backend,
            "json_count": len(self.items),
            "previous_chroma_count": 0,
            "deleted_from_chroma": 0,
            "rebuilt_count": 0,
            "skipped_count": 0,
            "failed_records": [],
            "backup_path": "",
            "collection_count": 0,
        }
        if self.backend != "chromadb" or self.client is None:
            self._init_chromadb()
        if self.collection is None:
            result["backend"] = self.backend
            result["failed_records"].append({"item_id": "", "reason": "ChromaDB 未初始化，无法重建"})
            return result

        if backup_json and self.json_path.exists():
            backup_path = self.json_path.with_name(
                f"{self.json_path.stem}.backup.{datetime.now().strftime('%Y%m%d%H%M%S')}{self.json_path.suffix}"
            )
            shutil.copy2(self.json_path, backup_path)
            result["backup_path"] = str(backup_path)

        existing_ids = self._chroma_ids()
        result["previous_chroma_count"] = len(existing_ids)
        try:
            for start in range(0, len(existing_ids), 500):
                batch = existing_ids[start : start + 500]
                if batch:
                    self.collection.delete(ids=batch)
                    result["deleted_from_chroma"] += len(batch)
        except Exception as exc:
            print(f"[ChromaDB] 批量删除旧索引失败，尝试重建 collection：{exc}", flush=True)
            try:
                self.client.delete_collection(name=self.collection_name)
            except Exception as delete_exc:
                print(f"[ChromaDB] delete_collection 失败，继续尝试覆盖写入：{delete_exc}", flush=True)
            self.collection = self._open_chroma_collection()

        for item_id, record in self.items.items():
            try:
                self._add_record_to_chroma(item_id, record)
                result["rebuilt_count"] += 1
            except Exception as exc:
                result["skipped_count"] += 1
                result["failed_records"].append({"item_id": item_id, "reason": str(exc)})

        if self.client is not None and hasattr(self.client, "persist"):
            try:
                self.client.persist()
            except Exception as exc:
                print(f"[ChromaDB] persist 失败：{exc}", flush=True)
        result["backend"] = self.backend
        result["collection_count"] = self.get_collection_count()
        print(
            f"[ChromaDB] 重建完成 collection={self.collection_name}, "
            f"json={result['json_count']}, rebuilt={result['rebuilt_count']}, skipped={result['skipped_count']}",
            flush=True,
        )
        return result

    def inspect_index_consistency(self) -> dict:
        chroma_ids = set(self._chroma_ids())
        json_ids = set(self.items.keys())
        return {
            "collection": self.collection_name,
            "backend": self.backend,
            "json_count": len(json_ids),
            "chroma_count": len(chroma_ids),
            "missing_in_chroma": sorted(json_ids - chroma_ids),
            "extra_in_chroma": sorted(chroma_ids - json_ids),
        }

    def build_query_vector(self, query_text: Optional[str] = None, query_image_path: Optional[str] = None) -> list[float]:
        if query_text and query_image_path:
            return fuse_vectors(
                self.encoder.encode_image(query_image_path),
                self.encoder.encode_text(query_text),
                image_weight=0.5,
                text_weight=0.5,
            )
        if query_text:
            return self.encoder.encode_text(query_text)
        if query_image_path:
            return self.encoder.encode_image(query_image_path)
        raise ValueError("必须提供 query_text 或 query_image_path")

    def _passes_prefilter(
        self,
        metadata: dict,
        location_filter: Optional[str],
        status_filter: Optional[str],
        max_age_days: Optional[int],
    ) -> bool:
        if location_filter and metadata.get("location") != location_filter:
            return False
        if status_filter and metadata.get("status") != status_filter:
            return False
        if max_age_days is not None:
            try:
                created = datetime.fromisoformat(str(metadata.get("created_at")))
                if created < datetime.now() - timedelta(days=max_age_days + 2):
                    return False
            except ValueError:
                pass
        return True

    def _search_json(
        self,
        query_vector: list[float],
        top_k: int,
        location_filter: Optional[str],
        status_filter: Optional[str],
        max_age_days: Optional[int],
    ) -> list[tuple[float, str, dict]]:
        rows = []
        for item_id, record in self.items.items():
            metadata = record["metadata"]
            if self._passes_prefilter(metadata, location_filter, status_filter, max_age_days):
                rows.append((cosine_similarity(query_vector, record["vectors"]["fused"]), item_id, metadata))
        rows.sort(key=lambda row: row[0], reverse=True)
        return rows[:top_k]

    def _search_chroma(
        self,
        query_vector: list[float],
        top_k: int,
        location_filter: Optional[str],
        status_filter: Optional[str],
        max_age_days: Optional[int],
    ) -> list[tuple[float, str, dict]]:
        if self.collection is None:
            return self._search_json(query_vector, top_k, location_filter, status_filter, max_age_days)
        count = self.get_collection_count()
        if count <= 0:
            return self._search_json(query_vector, top_k, location_filter, status_filter, max_age_days)
        try:
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=min(count, 100),
                include=["metadatas", "distances"],
            )
        except Exception as exc:
            print(f"[ChromaDB] 查询索引不可用，回退 JSON 备份：{exc}")
            return self._search_json(query_vector, top_k, location_filter, status_filter, max_age_days)
        rows = []
        for item_id, distance, metadata in zip(
            results.get("ids", [[]])[0],
            results.get("distances", [[]])[0],
            results.get("metadatas", [[]])[0],
        ):
            local_record = self.items.get(item_id)
            if not local_record:
                continue
            metadata = local_record.get("metadata", {})
            fused_vector = local_record.get("vectors", {}).get("fused")
            local_score = cosine_similarity(query_vector, fused_vector) if fused_vector else 1 - float(distance)
            chroma_score = 1 - float(distance)
            score = max(local_score, chroma_score)
            score = max(0.0, min(1.0, score))
            if self._passes_prefilter(metadata, location_filter, status_filter, max_age_days):
                rows.append((score, item_id, metadata))
        rows.sort(key=lambda row: row[0], reverse=True)
        expected_min = min(3, top_k)
        if len(rows) >= expected_min:
            return rows[:top_k]

        # Chroma can occasionally be out of sync with the JSON mirror after
        # bulk deletes or schema changes. Keep user-facing recall available by
        # merging persisted fused vectors back into the result set.
        print("[ChromaDB] 有效查询结果不足，补充 JSON 向量备份。")
        merged = {item_id: (score, item_id, metadata) for score, item_id, metadata in rows}
        for score, item_id, metadata in self._search_json(query_vector, top_k, location_filter, status_filter, max_age_days):
            previous = merged.get(item_id)
            if previous is None or score > previous[0]:
                merged[item_id] = (score, item_id, metadata)
        rows = sorted(merged.values(), key=lambda row: row[0], reverse=True)
        return rows[:top_k]

    def _search_once(
        self,
        query_vector: list[float],
        top_k: int,
        location_filter: Optional[str],
        status_filter: Optional[str],
        max_age_days: Optional[int],
    ) -> list[tuple[float, str, dict]]:
        if self.backend == "chromadb":
            return self._search_chroma(query_vector, top_k, location_filter, status_filter, max_age_days)
        return self._search_json(query_vector, top_k, location_filter, status_filter, max_age_days)

    def search_item(
        self,
        query_text: Optional[str] = None,
        query_image_path: Optional[str] = None,
        top_k: int = 5,
        location_filter: Optional[str] = None,
        status_filter: Optional[str] = "待认领",
        max_age_days: Optional[int] = 7,
    ) -> dict:
        query_vector = self.build_query_vector(query_text, query_image_path)
        fallback = None
        rows = self._search_once(query_vector, top_k, location_filter, status_filter, max_age_days)
        if len(rows) < min(3, top_k) and max_age_days is not None:
            rows = self._search_once(query_vector, top_k, location_filter, status_filter, None)
            fallback = "已放开时间约束"
        if len(rows) < min(3, top_k) and location_filter:
            rows = self._search_once(query_vector, top_k, None, status_filter, None)
            fallback = "已放开全部约束，建议人工核验地点"

        return {
            "ids": [item_id for _, item_id, _ in rows],
            "scores": [round(score, 4) for score, _, _ in rows],
            "distances": [round(1 - score, 4) for score, _, _ in rows],
            "metadatas": [metadata for _, _, metadata in rows],
            "query_vector": query_vector,
            "fallback": fallback,
            "prefilter": {"location": location_filter, "status": status_filter, "max_age_days": max_age_days},
        }

    def search_vector(
        self,
        query_vector: list[float],
        top_k: int = 5,
        location_filter: Optional[str] = None,
        status_filter: Optional[str] = "待认领",
        max_age_days: Optional[int] = 7,
    ) -> dict:
        """Search with a stored fused vector, avoiding encoder drift during historical rematch."""
        fallback = None
        rows = self._search_once(query_vector, top_k, location_filter, status_filter, max_age_days)
        if len(rows) < min(3, top_k) and max_age_days is not None:
            rows = self._search_once(query_vector, top_k, location_filter, status_filter, None)
            fallback = "已放开时间约束"
        if len(rows) < min(3, top_k) and location_filter:
            rows = self._search_once(query_vector, top_k, None, status_filter, None)
            fallback = "已放开全部约束，建议人工核验地点"

        return {
            "ids": [item_id for _, item_id, _ in rows],
            "scores": [round(score, 4) for score, _, _ in rows],
            "distances": [round(1 - score, 4) for score, _, _ in rows],
            "metadatas": [metadata for _, _, metadata in rows],
            "query_vector": query_vector,
            "fallback": fallback,
            "prefilter": {"location": location_filter, "status": status_filter, "max_age_days": max_age_days},
        }

    def delete_item(self, item_id: str) -> None:
        self.items.pop(item_id, None)
        self._save_json_items()
        if self.collection is not None:
            try:
                self.collection.delete(ids=[item_id])
            except Exception:
                pass

    def clear_all_items(self) -> int:
        """Remove all records from the local JSON mirror and vector collection."""
        removed_count = len(self.items)
        item_ids = list(self.items.keys())
        self.items = {}
        self._save_json_items()
        if self.collection is not None:
            try:
                chroma_ids = item_ids
                if hasattr(self.collection, "get"):
                    try:
                        chroma_ids = list(self.collection.get().get("ids", [])) or item_ids
                    except Exception:
                        chroma_ids = item_ids
                for start in range(0, len(chroma_ids), 500):
                    batch = chroma_ids[start : start + 500]
                    if batch:
                        self.collection.delete(ids=batch)
                if self.client is not None and hasattr(self.client, "persist"):
                    self.client.persist()
            except Exception as exc:
                print(f"[VectorStore] 清空 ChromaDB 集合失败，仅 JSON 已清空：{exc}")
        return removed_count

    def get_item(self, item_id: str) -> Optional[dict]:
        record = self.items.get(item_id)
        if not record:
            return None
        return {"item_id": item_id, "metadata": record.get("metadata", {}), "vectors": record.get("vectors", {})}

    def update_item_metadata(self, item_id: str, updates: dict) -> bool:
        record = self.items.get(item_id)
        if not record:
            return False
        metadata = {**record.get("metadata", {}), **updates, "updated_at": datetime.now().isoformat(timespec="seconds")}
        image_path = metadata.get("image_path", "")
        vectors = record.get("vectors", {})
        self.add_item(
            item_id=item_id,
            image_path=image_path,
            metadata=metadata,
            image_vector=vectors.get("image"),
            text_vector=vectors.get("text"),
            fused_vector=vectors.get("fused"),
        )
        return True

    def list_items(self, limit: Optional[int] = None) -> list[dict]:
        rows = []
        for item_id, record in self.items.items():
            rows.append(
                {
                    "item_id": item_id,
                    "metadata": record.get("metadata", {}),
                }
            )
        closed_statuses = {"已认领", "已找回"}
        rows.sort(key=lambda row: str(row["metadata"].get("created_at", "")), reverse=True)
        rows.sort(key=lambda row: row["metadata"].get("status") in closed_statuses)
        return rows[:limit] if limit else rows

    def find_by_metadata(self, key: str, value: str) -> Optional[dict]:
        if value in [None, ""]:
            return None
        for item_id, record in self.items.items():
            metadata = record.get("metadata", {})
            if metadata.get(key) == value:
                return {"item_id": item_id, "metadata": metadata}
        return None

    def get_collection_count(self) -> int:
        if self.collection is not None:
            try:
                count = int(self.collection.count())
                return count if count > 0 else len(self.items)
            except Exception:
                pass
        return len(self.items)

    def clear(self) -> None:
        self.items = {}
        self._save_json_items()
        if self.collection is not None:
            shutil.rmtree(self.persist_directory, ignore_errors=True)
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self._init_chromadb()
