"""
explore_hf_datasets.py — Previsualiza datasets de Hugging Face para evaluar
si sirven para Valentina.

Uso:
    python explore_hf_datasets.py
"""

from datasets import load_dataset


def show(dataset_id: str, n: int = 3, split: str = "train", config: str | None = None):
    label = f"{dataset_id}" + (f" ({config})" if config else "")
    print(f"\n{'='*70}\n  📦 {label}\n{'='*70}")
    try:
        kwargs = {"split": split, "streaming": True}
        if config:
            kwargs["name"] = config
        ds = load_dataset(dataset_id, **kwargs)
        for i, row in enumerate(ds):
            if i >= n:
                break
            print(f"\n--- Ejemplo {i+1} ---")
            for k, v in row.items():
                value = str(v).replace("\n", " ")[:180]
                print(f"  {k}: {value}")
    except Exception as exc:
        print(f"  ⚠️  Error: {exc}")


if __name__ == "__main__":
    # 1. Customer service estructurado (INGLÉS, útil para ver el formato)
    show("bitext/Bitext-customer-support-llm-chatbot-training-dataset", n=2)

    # 2. MASSIVE — Amazon, ESPAÑOL, intent classification multi-dominio
    show("AmazonScience/massive", n=4, config="es-ES")

    # 3. Amazon Reviews español (texto coloquial)
    show("mteb/amazon_reviews_multi", n=2, config="es")
