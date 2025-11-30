import os
import json
import math

import numpy as np
from flask import Flask, render_template, request, jsonify


# ==========================
#  Üçgen üyelik fonksiyonu
# ==========================

def tri_mf(x, a, b, c):
    """
    Üçgen üyelik fonksiyonu.
    a, b, c: taban ve tepe noktaları.
    """
    if a == b and b == c:
        return 0.0

    if x <= a or x >= c:
        return 0.0
    elif a < x < b:
        return (x - a) / (b - a) if b != a else 0.0
    elif b < x < c:
        return (c - x) / (c - b) if c != b else 0.0
    else:
        # x == b
        return 1.0


# ==========================
#  Basit bulanık talep modeli
# ==========================

class FuzzyDemandModel:
    PRICE_TERMS = ["low", "medium", "high"]
    COMP_TERMS = ["low", "medium", "high"]

    def __init__(self, price_mfs, comp_mfs, qty_min, qty_max, default_qty):
        self.price_mfs = price_mfs
        self.comp_mfs = comp_mfs
        self.qty_min = qty_min
        self.qty_max = qty_max
        self.default_qty = default_qty

        # 3 x 3 = 9 kural (price x competitor)
        self.rules = []
        for p in self.PRICE_TERMS:
            for c in self.COMP_TERMS:
                self.rules.append((p, c))

    @classmethod
    def from_params(cls, params: dict):
        return cls(
            price_mfs=params["price_mfs"],
            comp_mfs=params["comp_mfs"],
            qty_min=params["qty_min"],
            qty_max=params["qty_max"],
            default_qty=params["default_qty"],
        )

    def fuzzify_price(self, price: float):
        return {
            name: tri_mf(price, *params)
            for name, params in self.price_mfs.items()
        }

    def fuzzify_comp(self, comp_price: float):
        return {
            name: tri_mf(comp_price, *params)
            for name, params in self.comp_mfs.items()
        }

    def infer_quantity(self, price: float, comp_price: float, rule_outputs):
        """
        Verilen fiyat ve rakip fiyatına göre beklenen talebi (adet) döndürür.
        rule_outputs: Genetik algoritmanın bulduğu 9 adet kural çıktısı.
        """
        mu_price = self.fuzzify_price(price)
        mu_comp = self.fuzzify_comp(comp_price)

        num = 0.0
        den = 0.0

        for idx, (p_term, c_term) in enumerate(self.rules):
            w = mu_price[p_term] * mu_comp[c_term]
            if w <= 0:
                continue
            num += w * float(rule_outputs[idx])
            den += w

        if den == 0.0:
            # Hiçbir kural aktif değilse, GA tarafından öğrenilen
            # ortalama / default talep değerine geri düşüyoruz.
            return float(self.default_qty)

        return num / den


# ==========================
#  Flask uygulaması ayarları
# ==========================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(ROOT_DIR, "models", "model_params.json")

with open(MODEL_PATH, "r", encoding="utf-8") as f:
    model_params = json.load(f)

# Bulanık model
fuzzy_model = FuzzyDemandModel.from_params(model_params)

# GA çıktıları
BEST_RULES = np.array(model_params["best_rules"], dtype=float)

# Eğitim aralıkları
PRICE_MIN = float(model_params["price_min"])
PRICE_MAX = float(model_params["price_max"])
COMP_MIN = float(model_params["comp_min"])
COMP_MAX = float(model_params["comp_max"])

QTY_MIN = float(model_params["qty_min"])
QTY_MAX = float(model_params["qty_max"])
DEFAULT_QTY = float(model_params["default_qty"])

MSE = float(model_params["mse"])


def compute_grid(comp_price: float, p_min: float, p_max: float, n_points: int = 60):
    """
    Verilen rakip fiyatı için [p_min, p_max] aralığında
    beklenen talep ve gelir eğrilerini hesaplar.
    """
    prices = np.linspace(p_min, p_max, n_points)
    qty_list = []
    rev_list = []

    for p in prices:
        q = fuzzy_model.infer_quantity(float(p), float(comp_price), BEST_RULES)
        qty_list.append(q)
        rev_list.append(p * q)

    return prices, qty_list, rev_list


def _parse_optional_float(val):
    """
    "" / None -> None
    aksi halde virgül-nokta normalize edip float'a çevirir, hata varsa None döner.
    """
    if val in (None, ""):
        return None
    try:
        return float(str(val).replace(",", "."))
    except (TypeError, ValueError):
        return None


# ==========================
#  Rotalar
# ==========================

@app.route("/")
def index():
    """
    Ana sayfa: Model özetini ve formu gösterir.
    """
    return render_template(
        "index.html",
        price_min=PRICE_MIN,
        price_max=PRICE_MAX,
        comp_min=COMP_MIN,
        comp_max=COMP_MAX,
        qty_min=QTY_MIN,
        qty_max=QTY_MAX,
        default_qty=DEFAULT_QTY,
        rule_count=len(BEST_RULES),
        mse=MSE,
    )


@app.route("/api/recommend_price", methods=["POST"])
def recommend_price():
    """
    Kullanıcıdan:
      - competitor_price  (zorunlu)
      - price_min, price_max (opsiyonel)
    alır ve bu aralıkta beklenen geliri maksimize eden fiyatı bulur.

    Mantık:
      * Kullanıcı aralık girerse:
          - Önce alt/üst sınır küçükten büyüğe SIRALANIR.
          - Daha sonra eğitim aralığı [PRICE_MIN, PRICE_MAX] ile KESİŞİM alınır.
          - Kesişim boşsa tam eğitim aralığına geri dönülür.
      * Hiç aralık girmezse:
          - Rakip fiyat etrafında ±40 TL pencere alınır
            ve eğitim aralığı ile kesiştirilir.
    """
    data = request.get_json(force=True) or {}

    # 1) Rakip fiyatı oku
    comp_price_raw = data.get("competitor_price")
    try:
        comp_price = float(str(comp_price_raw).replace(",", "."))
    except (TypeError, ValueError):
        comp_price = None

    # 2) Kullanıcı alt/üst sınırları oku
    user_min = _parse_optional_float(data.get("price_min"))
    user_max = _parse_optional_float(data.get("price_max"))

    # 3) Rakip fiyatı doğrula / clamp et
    if comp_price is None or not math.isfinite(comp_price) or comp_price <= 0:
        # Geçersizse eğitim fiyat aralığının ortasını kullan
        comp_price = 0.5 * (PRICE_MIN + PRICE_MAX)

    # Rakip fiyatı eğitim aralığına sıkıştır (comp_min, comp_max)
    comp_price = max(min(comp_price, COMP_MAX), COMP_MIN)

    # 4) Kullanıcı aralığı varsa sıralama (sıralama mantığı BURADA)
    user_range_swapped = False
    if (
        user_min is not None
        and math.isfinite(user_min)
        and user_max is not None
        and math.isfinite(user_max)
        and user_min > user_max
    ):
        # Alt > üst ise yer değiştir
        user_min, user_max = user_max, user_min
        user_range_swapped = True

    # JS tarafına geri göstermek için normalize edilmiş kullanıcı aralığını sakla
    resp_user_min = user_min
    resp_user_max = user_max

    # 5) Kullanıcı aralığı gerçekten var mı?
    has_user_range = (
        (user_min is not None and math.isfinite(user_min))
        or (user_max is not None and math.isfinite(user_max))
    )

    if has_user_range:
        # Kullanıcının aralığı + eğitim aralığı kesişimi
        eff_min = PRICE_MIN
        eff_max = PRICE_MAX

        if user_min is not None and math.isfinite(user_min):
            eff_min = max(eff_min, user_min)

        if user_max is not None and math.isfinite(user_max):
            eff_max = min(eff_max, user_max)

        # Kesişim boşsa, eğitim aralığına geri dön
        if eff_min >= eff_max:
            eff_min = PRICE_MIN
            eff_max = PRICE_MAX
    else:
        # Kullanıcı aralık vermediyse: Rakip ±40 TL penceresi
        SEARCH_HALF_WIDTH = 40.0
        eff_min = max(PRICE_MIN, comp_price - SEARCH_HALF_WIDTH)
        eff_max = min(PRICE_MAX, comp_price + SEARCH_HALF_WIDTH)

    # 6) Grid hesapla
    prices, qty_list, rev_list = compute_grid(comp_price, eff_min, eff_max)

    if len(prices) == 0:
        return jsonify(
            {
                "status": "error",
                "message": "Geçerli fiyat aralığı bulunamadı.",
            }
        ), 400

    best_idx = int(np.argmax(rev_list))
    best_price = float(prices[best_idx])
    best_qty = float(qty_list[best_idx])
    best_revenue = float(rev_list[best_idx])

    # 7) JSON cevap – script.js bu alan adlarını bekliyor
    response = {
        "status": "ok",
        "message": "Dinamik fiyat önerisi başarıyla hesaplandı.",
        # Modelin gördüğü eğitim fiyat aralığı
        "model_price_min": float(PRICE_MIN),
        "model_price_max": float(PRICE_MAX),
        # Bu senaryoda gerçekten taranan aralık
        "effective_min": float(eff_min),
        "effective_max": float(eff_max),
        # Girdi olarak kullanılan (clamp edilmiş) rakip fiyat
        "competitor_price": float(comp_price),
        # En iyi fiyat ve bu fiyattaki beklenen talep / gelir
        "best_price": round(best_price, 2),
        "expected_qty": round(best_qty, 2),
        "expected_revenue": round(best_revenue, 2),
        # Grafikler için grid
        "price_grid": [round(float(p), 2) for p in prices],
        "qty_grid": [round(float(q), 2) for q in qty_list],
        "revenue_grid": [round(float(r), 2) for r in rev_list],
        # Kullanıcının (normalize edilmiş) girdiği aralık
        "user_price_min": None if resp_user_min is None else float(resp_user_min),
        "user_price_max": None if resp_user_max is None else float(resp_user_max),
        "user_range_swapped": bool(user_range_swapped),
    }

    return jsonify(response)


if __name__ == "__main__":
    # Geliştirme sunucusu
    app.run(host="127.0.0.1", port=5000, debug=True)
