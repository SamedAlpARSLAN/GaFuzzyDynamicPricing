"""
GA + Fuzzy Logic ile Dinamik Fiyatlandırma Modeli
Veri seti: Kaggle - Retail Price Optimization (retail_price.csv)

Bu script:
  - Seçilen kategori için veriyi okur
  - Fiyat ve rakip fiyatı için bulanık üyelik fonksiyonları tanımlar
  - 9 kuralın çıktı sabitlerini Genetik Algoritma ile öğrenir
  - Konsola örnek sonuçlar yazar
  - Öğrenilen model parametrelerini models/model_params.json dosyasına kaydeder
"""

import os
import json
import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


# -------------------------
# 1. Yardımcı fonksiyonlar
# -------------------------

def tri_mf(x: float, a: float, b: float, c: float) -> float:
    """
    Üçgen üyelik fonksiyonu.
    a <= b <= c olmalı.
    """
    if a == b and b == c:
        return 0.0
    if x <= a or x >= c:
        return 0.0
    elif a < x < b:
        return (x - a) / (b - a) if b != a else 0.0
    elif b < x < c:
        return (c - x) / (c - b) if c != b else 0.0
    else:  # x == b
        return 1.0


class FuzzyDemandModel:
    """
    2 girişli (fiyat, rakip ortalama fiyat) 1 çıkışlı (talep) bulanık sistem.
    Girişler için üyelik fonksiyonları sabit, kural çıktıları (9 adet sabit)
    GA tarafından öğreniliyor.
    """

    PRICE_TERMS = ["low", "medium", "high"]
    COMP_TERMS = ["low", "medium", "high"]

    def __init__(
        self,
        price_mfs: dict,
        comp_mfs: dict,
        qty_min: float,
        qty_max: float,
        default_qty: float,
    ):
        """
        price_mfs / comp_mfs:
            {
              "low":   (a, b, c),
              "medium":(a, b, c),
              "high":  (a, b, c)
            }
        qty_min, qty_max: GA için sınırlar
        default_qty: hiç kural ateşlenmezse kullanılacak değer (ör: ortalama talep)
        """
        self.price_mfs = price_mfs
        self.comp_mfs = comp_mfs
        self.qty_min = qty_min
        self.qty_max = qty_max
        self.default_qty = default_qty

        # 3x3 = 9 kural kombinasyonu (PRICE x COMP)
        self.rules: List[Tuple[str, str]] = []
        for p in self.PRICE_TERMS:
            for c in self.COMP_TERMS:
                self.rules.append((p, c))

    # --- fuzzification ---

    def fuzzify_price(self, price: float) -> dict:
        return {
            name: tri_mf(price, *params)
            for name, params in self.price_mfs.items()
        }

    def fuzzify_comp(self, comp_price: float) -> dict:
        return {
            name: tri_mf(comp_price, *params)
            for name, params in self.comp_mfs.items()
        }

    # --- inference ---

    def infer_quantity(self, price: float, comp_price: float, rule_outputs: np.ndarray) -> float:
        """
        rule_outputs: uzunluğu 9 olan numpy array, her kuralın çıktısı (sabit talep seviyesi).
        Çıktı: Sugeno tipi ağırlıklı ortalama.
        """
        mu_price = self.fuzzify_price(price)
        mu_comp = self.fuzzify_comp(comp_price)

        num = 0.0
        den = 0.0

        for idx, (p_term, c_term) in enumerate(self.rules):
            w = mu_price[p_term] * mu_comp[c_term]
            if w > 0.0:
                num += w * float(rule_outputs[idx])
                den += w

        if den == 0.0:
            # Güvenli tarafta kal: hiç kural ateşlenmezse default talep
            return float(self.default_qty)
        return num / den

    def batch_predict(self, prices: np.ndarray, comps: np.ndarray, rule_outputs: np.ndarray) -> np.ndarray:
        preds = np.zeros_like(prices, dtype=float)
        for i, (p, c) in enumerate(zip(prices, comps)):
            preds[i] = self.infer_quantity(float(p), float(c), rule_outputs)
        return preds


# -------------------------
# 2. Genetik algoritma
# -------------------------

@dataclass
class GAConfig:
    pop_size: int = 40
    n_generations: int = 80
    crossover_rate: float = 0.8
    mutation_rate: float = 0.2
    elite_frac: float = 0.2
    mutation_std_frac: float = 0.1  # qty aralığının yüzdesi
    random_seed: int = 42


class GeneticAlgorithm:
    def __init__(
        self,
        model: FuzzyDemandModel,
        prices: np.ndarray,
        comps: np.ndarray,
        qty_true: np.ndarray,
        config: GAConfig,
    ):
        self.model = model
        self.prices = prices
        self.comps = comps
        self.qty_true = qty_true
        self.cfg = config

        random.seed(self.cfg.random_seed)
        np.random.seed(self.cfg.random_seed)

        self.gene_min = self.model.qty_min
        self.gene_max = self.model.qty_max
        self.n_genes = 9  # 9 kural çıktısı

        self.mutation_std = (self.gene_max - self.gene_min) * self.cfg.mutation_std_frac

    # --- GA yardımcıları ---

    def _init_population(self) -> np.ndarray:
        """
        pop_size x n_genes boyutlu başlangıç popülasyonu.
        """
        return np.random.uniform(
            low=self.gene_min,
            high=self.gene_max,
            size=(self.cfg.pop_size, self.n_genes),
        )

    def _evaluate_individual(self, individual: np.ndarray) -> float:
        """
        Bireyin fitness'ı: -MSE (büyük olması iyi).
        """
        preds = self.model.batch_predict(self.prices, self.comps, individual)
        mse = float(np.mean((preds - self.qty_true) ** 2))
        return -mse

    def _evaluate_population(self, pop: np.ndarray) -> np.ndarray:
        fitness = np.zeros(pop.shape[0], dtype=float)
        for i, ind in enumerate(pop):
            fitness[i] = self._evaluate_individual(ind)
        return fitness

    def _select_parents(self, pop: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """
        Basit turnuva seçimi.
        """
        parents = []
        for _ in range(pop.shape[0]):
            i, j = np.random.randint(0, pop.shape[0], size=2)
            if fitness[i] > fitness[j]:
                parents.append(pop[i].copy())
            else:
                parents.append(pop[j].copy())
        return np.array(parents)

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() > self.cfg.crossover_rate:
            return parent1.copy(), parent2.copy()

        # uniform crossover
        mask = np.random.rand(self.n_genes) < 0.5
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        return child1, child2

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        for g in range(self.n_genes):
            if random.random() < self.cfg.mutation_rate:
                individual[g] += np.random.normal(0.0, self.mutation_std)
        # sınırla
        individual = np.clip(individual, self.gene_min, self.gene_max)
        return individual

    def run(self) -> Tuple[np.ndarray, List[float]]:
        pop = self._init_population()
        history_best = []

        elite_size = max(1, int(self.cfg.pop_size * self.cfg.elite_frac))

        for gen in range(self.cfg.n_generations):
            fitness = self._evaluate_population(pop)
            best_idx = int(np.argmax(fitness))
            best_fit = float(fitness[best_idx])
            history_best.append(best_fit)

            print(f"Generation {gen+1}/{self.cfg.n_generations} - Best fitness: {best_fit:.4f}")

            # elitleri sakla
            elite_indices = fitness.argsort()[-elite_size:]
            elites = pop[elite_indices]

            # ebeveyn seçimi
            parents = self._select_parents(pop, fitness)

            # yeni popülasyonu oluştur
            next_pop = []

            # elitleri doğrudan kopyala
            for e in elites:
                next_pop.append(e.copy())

            # geri kalanlar için crossover + mutasyon
            while len(next_pop) < self.cfg.pop_size:
                i, j = np.random.randint(0, parents.shape[0], size=2)
                p1 = parents[i]
                p2 = parents[j]
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1)
                if len(next_pop) < self.cfg.pop_size:
                    next_pop.append(c1)
                c2 = self._mutate(c2)
                if len(next_pop) < self.cfg.pop_size:
                    next_pop.append(c2)

            pop = np.array(next_pop)

        # son popülasyondan en iyiyi döndür
        fitness = self._evaluate_population(pop)
        best_idx = int(np.argmax(fitness))
        best_individual = pop[best_idx].copy()
        return best_individual, history_best

    # Ekstra: MSE hesaplamak için
    def evaluate_mse(self, individual: np.ndarray) -> float:
        preds = self.model.batch_predict(self.prices, self.comps, individual)
        mse = float(np.mean((preds - self.qty_true) ** 2))
        return mse


# -------------------------
# 3. Dinamik fiyat önerisi
# -------------------------

def recommend_price_for_row(
    model: FuzzyDemandModel,
    rule_outputs: np.ndarray,
    comp_price: float,
    price_min: float,
    price_max: float,
    n_grid: int = 50,
) -> Tuple[float, float, float]:
    """
    Verilen bir rakip fiyatına göre (comp_price), olası fiyat aralığında grid taraması yapar,
    beklenen geliri maksimize eden fiyatı döndürür.

    Dönüş:
        best_price, expected_qty, expected_revenue
    """
    prices = np.linspace(price_min, price_max, n_grid)
    best_price = prices[0]
    best_qty = model.infer_quantity(best_price, comp_price, rule_outputs)
    best_rev = best_price * best_qty

    for p in prices[1:]:
        q = model.infer_quantity(p, comp_price, rule_outputs)
        rev = p * q
        if rev > best_rev:
            best_price = p
            best_qty = q
            best_rev = rev

    return float(best_price), float(best_qty), float(best_rev)


# -------------------------
# 4. Ana akış
# -------------------------

def main():
    # 4.1. Veri setini oku
    df = pd.read_csv("data/retail_price.csv")

    # 4.2. Tek bir kategori seç (örnek: garden_tools)
    target_category = "garden_tools"
    df_cat = df[df["product_category_name"] == target_category].copy()

    if df_cat.empty:
        raise ValueError(
            f"{target_category} kategorisi veri setinde bulunamadı. "
            "Önce df['product_category_name'].unique() ile kategorileri kontrol edin."
        )

    # 4.3. İlgili sütunları seç
    needed_cols = [
        "unit_price",
        "qty",
        "comp_1",
        "comp_2",
        "comp_3",
    ]
    for col in needed_cols:
        if col not in df_cat.columns:
            raise ValueError(f"Gerekli sütun eksik: {col}")

    df_cat["comp_mean"] = df_cat[["comp_1", "comp_2", "comp_3"]].mean(axis=1)
    df_cat = df_cat.dropna(subset=["unit_price", "qty", "comp_mean"])

    # numpy dizilerine çevir
    prices = df_cat["unit_price"].to_numpy(dtype=float)
    comps = df_cat["comp_mean"].to_numpy(dtype=float)
    qty_true = df_cat["qty"].to_numpy(dtype=float)

    # 4.4. Üyelik fonksiyonu aralıkları (min, orta, max)
    price_min, price_max = float(prices.min()), float(prices.max())
    comp_min, comp_max = float(comps.min()), float(comps.max())
    qty_min, qty_max = float(qty_true.min()), float(qty_true.max())
    default_qty = float(qty_true.mean())

    price_mid = (price_min + price_max) / 2.0
    comp_mid = (comp_min + comp_max) / 2.0

    price_mfs = {
        "low": (price_min, price_min, price_mid),
        "medium": (price_min, price_mid, price_max),
        "high": (price_mid, price_max, price_max),
    }

    comp_mfs = {
        "low": (comp_min, comp_min, comp_mid),
        "medium": (comp_min, comp_mid, comp_max),
        "high": (comp_mid, comp_max, comp_max),
    }

    model = FuzzyDemandModel(
        price_mfs=price_mfs,
        comp_mfs=comp_mfs,
        qty_min=qty_min,
        qty_max=qty_max,
        default_qty=default_qty,
    )

    # 4.5. GA ayarları ve çalıştırma
    cfg = GAConfig(
        pop_size=40,
        n_generations=60,
        crossover_rate=0.8,
        mutation_rate=0.3,
        elite_frac=0.25,
        mutation_std_frac=0.15,
        random_seed=42,
    )

    ga = GeneticAlgorithm(
        model=model,
        prices=prices,
        comps=comps,
        qty_true=qty_true,
        config=cfg,
    )

    best_rules, history = ga.run()
    mse = ga.evaluate_mse(best_rules)
    print("\nEn iyi bireyin MSE değeri:", mse)

    # 4.6. Dinamik fiyat örnekleri
    print("\n--- Dinamik fiyat önerisi örnekleri ---")
    example_rows = df_cat.iloc[:3]

    for idx, row in example_rows.iterrows():
        base_price = float(row["unit_price"])
        comp_price = float(row["comp_mean"])
        qty_obs = float(row["qty"])

        best_price, best_qty, best_rev = recommend_price_for_row(
            model=model,
            rule_outputs=best_rules,
            comp_price=comp_price,
            price_min=price_min,
            price_max=price_max,
            n_grid=40,
        )

        base_rev = base_price * qty_obs

        print(f"\nKayıt ID: {idx}")
        print(f"  Mevcut fiyat: {base_price:.2f}, Gözlenen miktar: {qty_obs:.2f}, Gelir: {base_rev:.2f}")
        print(f"  Rakip ort. fiyatı: {comp_price:.2f}")
        print(f"  Önerilen dinamik fiyat: {best_price:.2f}")
        print(f"  Modelin bu fiyattaki beklenen talebi: {best_qty:.2f}")
        print(f"  Beklenen gelir: {best_rev:.2f}")

    # 4.7. Kuralları özetle
    print("\n--- Öğrenilen kural çıktıları (talep seviyeleri) ---")
    for i, (p_term, c_term) in enumerate(model.rules):
        print(f"Kural {i+1}: IF price is {p_term} AND competitor is {c_term} THEN qty ~= {best_rules[i]:.2f}")

    # 4.8. Model parametrelerini JSON'a kaydet
    model_data = {
        "price_mfs": {k: list(v) for k, v in price_mfs.items()},
        "comp_mfs": {k: list(v) for k, v in comp_mfs.items()},
        "qty_min": qty_min,
        "qty_max": qty_max,
        "default_qty": default_qty,
        "price_min": price_min,
        "price_max": price_max,
        "comp_min": comp_min,
        "comp_max": comp_max,
        "best_rules": best_rules.tolist(),
        "mse": mse,
    }

    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "model_params.json")
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(model_data, f, ensure_ascii=False, indent=2)

    print(f'\nModel parametreleri "{model_path}" dosyasına kaydedildi.')


if __name__ == "__main__":
    main()
