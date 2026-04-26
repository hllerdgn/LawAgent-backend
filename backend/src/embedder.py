"""
embedder.py  —  LawAgent Mursit Embedder (v10-Production)
----------------------------------------------------------
Değişiklikler v9 → v10:
  - encode_single() metodu eklendi (retrieval.py monkey-patch kaldırıldı)
  - Quantize path mantığı sadeleştirildi ve hata toleranslı hale getirildi
  - __init__ süresi print'i düzeltildi
  - QDRANT_STORAGE path env var'dan da okunabiliyor
"""

import argparse
import json
import os
import time
import uuid

import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# ─── PATHS ────────────────────────────────────────────────────────────────────

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, "data")
MODEL_NAME    = "newmindai/Mursit-Base-TR-Retrieval"
COLLECTION_NAME = "lawagent_mursit"
CHUNK_CORPUS  = os.path.join(DATA_DIR, "chunk_corpus.json")
QDRANT_STORAGE = os.environ.get(
    "QDRANT_STORAGE",
    "/content/drive/MyDrive/lawagent/backend/src/data/qdrant_storage",
)
QUANTIZE_PATH = os.path.join(DATA_DIR, "mursit_int8.pt")
BATCH_SIZE    = 32
DISTANCE_METRIC = qmodels.Distance.COSINE


# ─── EMBEDDER ─────────────────────────────────────────────────────────────────

class MursitEmbedder:
    """
    Mursit-Base-TR-Retrieval için embedding sınıfı.

    quantize=False → float32  (~594 MB RAM, ~145 ms/sorgu)
    quantize=True  → int8     (~173 MB RAM,  ~94 ms/sorgu,  %3 MRR kaybı)
    """

    def __init__(self, quantize: bool = False):
        self.quantize = quantize
        self.device   = "cpu"

        fmt = "int8" if quantize else "float32"
        print(f"[Mursit] Model yükleniyor ({fmt})...")
        t0 = time.time()

        self.st = SentenceTransformer(MODEL_NAME, device=self.device)
        self.vector_size = self.st.get_embedding_dimension()

        if quantize:
            self._load_or_quantize()

        print(f"[Mursit] Hazır — {time.time()-t0:.1f}s | dim={self.vector_size}")

    # ------------------------------------------------------------------

    def _load_or_quantize(self) -> None:
        transformer_module = self.st._first_module().auto_model

        quantized = torch.quantization.quantize_dynamic(
            transformer_module, {torch.nn.Linear}, dtype=torch.qint8
        )

        if os.path.exists(QUANTIZE_PATH):
            print("[Mursit] Kaydedilmiş int8 ağırlıklar yükleniyor...")
            try:
                state = torch.load(QUANTIZE_PATH, map_location="cpu")
                quantized.load_state_dict(state)
            except Exception as e:
                print(f"[UYARI] Kaydedilmiş int8 yüklenemedi, sıfırdan quantize edildi: {e}")
        else:
            print("[Mursit] Quantize ediliyor (ilk kez)...")
            os.makedirs(os.path.dirname(QUANTIZE_PATH) or ".", exist_ok=True)
            torch.save(quantized.state_dict(), QUANTIZE_PATH)
            print(f"[Mursit] int8 kaydedildi → {QUANTIZE_PATH}")

        self.st._first_module().auto_model = quantized

    # ------------------------------------------------------------------

    def encode(self, texts: list[str], normalize: bool = True) -> list:
        """Liste halinde metinleri vektöre çevirir → Python list of list[float]."""
        return self.st.encode(
            texts,
            normalize_embeddings=normalize,
            show_progress_bar=False,
            batch_size=BATCH_SIZE,
        ).tolist()

    def encode_single(self, text: str, normalize: bool = True) -> list[float]:
        """Sorguyu Mursit formatına uygun prefix ile vektöre çevirir."""
        prefix = "query: "
        full_text = prefix + text.strip() # Boşlukları temizle
        
        return self.st.encode(
            full_text, 
            normalize_embeddings=normalize,
            show_progress_bar=False,
            convert_to_numpy=True 
        ).tolist()

    def kaydet(self, yol: str = QUANTIZE_PATH) -> None:
        if not self.quantize:
            print("[UYARI] float32 model kaydedilmiyor. --quantize ile çalıştır.")
            return
        os.makedirs(os.path.dirname(yol) or ".", exist_ok=True)
        torch.save(self.st._first_module().auto_model.state_dict(), yol)
        mb = os.path.getsize(yol) / 1024 / 1024
        print(f"[Mursit] int8 kaydedildi → {yol} ({mb:.1f} MB)")


# ─── QDRANT HELPERS ───────────────────────────────────────────────────────────

def _chunk_id_to_uint64(cid: str) -> int:
    return uuid.uuid5(uuid.NAMESPACE_DNS, str(cid)).int >> 64


def _get_existing_ids(client: QdrantClient, corpus: list) -> set:
    ids = [_chunk_id_to_uint64(c["chunk_id"]) for c in corpus]
    existing = set()
    for i in range(0, len(ids), 1000):
        try:
            points = client.retrieve(
                collection_name=COLLECTION_NAME,
                ids=ids[i : i + 1000],
                with_payload=False,
                with_vectors=False,
            )
            existing.update(p.id for p in points)
        except Exception:
            pass
    return existing


def _ensure_collection(
    client: QdrantClient, vector_size: int, reset: bool
) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if reset and COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"[Qdrant] Collection silindi: {COLLECTION_NAME}")
        existing.remove(COLLECTION_NAME)
    if COLLECTION_NAME not in existing:
        client.create_collection(
            COLLECTION_NAME,
            vectors_config=qmodels.VectorParams(
                size=vector_size, distance=DISTANCE_METRIC
            ),
        )
        print(f"[Qdrant] Collection oluşturuldu: {COLLECTION_NAME}")
    else:
        count = client.count(COLLECTION_NAME).count
        print(f"[Qdrant] Mevcut: {COLLECTION_NAME} ({count} kayıt)")


# ─── EMBED CORPUS ─────────────────────────────────────────────────────────────

def embed_corpus(
    reset: bool = False,
    test_mode: bool = False,
    quantize: bool = False,
) -> None:
    if not os.path.exists(CHUNK_CORPUS):
        print(f"[HATA] {CHUNK_CORPUS} bulunamadı. Önce legal_chunker.py çalıştır.")
        return

    with open(CHUNK_CORPUS, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    if test_mode:
        corpus = corpus[:20]
        print("[Test] Sadece ilk 20 chunk işlenecek.")

    embedder = MursitEmbedder(quantize=quantize)
    client   = QdrantClient(path=QDRANT_STORAGE)

    _ensure_collection(client, embedder.vector_size, reset)

    existing = set() if reset else _get_existing_ids(client, corpus)
    yeni     = [c for c in corpus if _chunk_id_to_uint64(c["chunk_id"]) not in existing]

    print(
        f"[Embed] Toplam={len(corpus)} | Mevcut={len(existing)} | Eklenecek={len(yeni)}"
    )
    if not yeni:
        print("[Embed] Her şey güncel, işlem yok.")
        return

    t0, eklenen, toplam = time.time(), 0, len(yeni)

    for i in range(0, toplam, BATCH_SIZE):
        batch = yeni[i : i + BATCH_SIZE]
        enriched_texts = [
            f"{c.get('law', '')} Madde {c.get('article_no', '')}: {c.get('text', '')}"
            for idx, c in enumerate(batch)
        ]
        vecs  = embedder.encode(enriched_texts)

        points = [
            qmodels.PointStruct(
                id=_chunk_id_to_uint64(c["chunk_id"]),
                vector=vecs[idx],
                payload={
                    "chunk_id":   c.get("chunk_id", ""),
                    "text":       c.get("text", ""),
                    "law":        c.get("law", ""),
                    "article_no": c.get("article_no", ""),
                    "source":     c.get("source", ""),
                    "decision_id":c.get("decision_id", ""),
                    "token_len":  c.get("token_len", 0),
                },
            )
            for idx, c in enumerate(batch)
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points)

        eklenen += len(batch)
        elapsed  = time.time() - t0
        remaining = (elapsed / eklenen) * (toplam - eklenen) if eklenen else 0
        print(
            f"  {eklenen:4d}/{toplam} (%{eklenen/toplam*100:.0f})"
            f" | ~{remaining/60:.1f} dk kaldı"
        )

    print(f"\n[Embed] Tamamlandı. {eklenen} chunk eklendi. ({time.time()-t0:.1f}s)")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LawAgent Mursit Embedder v10")
    parser.add_argument("--reset",    action="store_true", help="Collection'ı sıfırla")
    parser.add_argument("--test",     action="store_true", help="İlk 20 chunk")
    parser.add_argument("--quantize", action="store_true", help="int8 quantization")
    parser.add_argument("--kaydet",   action="store_true", help="Quantized modeli kaydet")
    args = parser.parse_args()

    if args.quantize and args.kaydet:
        embedder = MursitEmbedder(quantize=True)
        embedder.kaydet()
    else:
        embed_corpus(reset=args.reset, test_mode=args.test, quantize=args.quantize)