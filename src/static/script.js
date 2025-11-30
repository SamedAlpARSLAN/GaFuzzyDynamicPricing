// ============ Chart.js genel ayar ============

if (window.Chart) {
  Chart.defaults.font.family =
    '"Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
  Chart.defaults.font.size = 12;
}

// ============ Yardımcı fonksiyonlar ============

function tlFormat(num) {
  // Sayıyı 2 haneli TL formatına çevir (virgüllü göster)
  if (typeof num !== "number" || !isFinite(num)) return "-";
  return num.toFixed(2).replace(".", ",");
}

function parseNumber(str) {
  if (str === null || str === undefined) return null;
  const cleaned = str.trim().replace(",", ".");
  if (cleaned === "") return null;
  const val = Number(cleaned);
  return Number.isNaN(val) ? NaN : val;
}

let demandChart = null;
let revenueChart = null;

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("price-form");
  const btn = document.getElementById("btnCalculate");
  const errorBox = document.getElementById("errorBox");
  const resultsSection = document.getElementById("resultsSection");
  const chartsSection = document.getElementById("chartsSection");

  function showError(msg) {
    errorBox.textContent = msg;
    errorBox.style.display = "block";
  }

  function clearError() {
    errorBox.textContent = "";
    errorBox.style.display = "none";
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    clearError();
    resultsSection.style.display = "none";
    chartsSection.style.display = "none";

    const compVal = parseNumber(
      document.getElementById("competitorPrice").value
    );
    if (!Number.isFinite(compVal) || compVal <= 0) {
      showError(
        "Rakip ortalama fiyatı için pozitif bir sayı giriniz (örneğin 80 veya 80,50)."
      );
      return;
    }

    const minInput = document.getElementById("priceMin").value;
    const maxInput = document.getElementById("priceMax").value;

    btn.disabled = true;
    btn.textContent = "Hesaplanıyor...";

    const payload = {
      competitor_price: compVal, // backend float olarak parse edecek
      price_min: minInput.trim(),
      price_max: maxInput.trim(),
    };

    try {
      const resp = await fetch("/api/recommend_price", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!resp.ok) {
        showError("Sunucudan geçerli bir yanıt alınamadı.");
        return;
      }

      let data;
      try {
        data = await resp.json();
      } catch (err) {
        console.error("JSON parse hatası:", err);
        showError("Sunucudan geçerli bir yanıt alınamadı.");
        return;
      }

      if (!data || data.status !== "ok") {
        showError(
          data && data.message
            ? data.message
            : "Sunucudan geçerli bir yanıt alınamadı."
        );
        return;
      }

      updateResults(data, resultsSection, chartsSection);
    } catch (err) {
      console.error("İstek hatası:", err);
      showError("Sunucudan geçerli bir yanıt alınamadı.");
    } finally {
      btn.disabled = false;
      btn.textContent = "Dinamik Fiyatı Hesapla";
    }
  });
});

function updateResults(data, resultsSection, chartsSection) {
  // 3.1 Sayısal sonuçlar
  document.getElementById("resCompetitorPrice").textContent = tlFormat(
    data.competitor_price
  );
  document.getElementById(
    "resModelRange"
  ).textContent = `${tlFormat(data.model_price_min)} – ${tlFormat(
    data.model_price_max
  )}`;
  document.getElementById(
    "resEffectiveRange"
  ).textContent = `${tlFormat(data.effective_min)} – ${tlFormat(
    data.effective_max
  )}`;
  document.getElementById("resBestPrice").textContent = tlFormat(
    data.best_price
  );
  document.getElementById("resExpectedQty").textContent = Number(
    data.expected_qty
  ).toFixed(2);
  document.getElementById("resExpectedRevenue").textContent = tlFormat(
    data.expected_revenue
  );

  // Kullanıcının girdiği (normalize edilmiş) aralık
  const userMin = data.user_price_min;
  const userMax = data.user_price_max;
  const swapped = !!data.user_range_swapped;

  let rangeSentence = "";

  if (userMin != null || userMax != null) {
    // Kullanıcı bir aralık girdi
    let typedDesc;
    if (userMin != null && userMax != null) {
      typedDesc = `${tlFormat(userMin)} – ${tlFormat(userMax)} TL aralığını`;
    } else if (userMin != null) {
      typedDesc = `${tlFormat(userMin)} TL ve üzerini`;
    } else {
      typedDesc = `${tlFormat(userMax)} TL ve altını`;
    }

    if (swapped) {
      rangeSentence =
        ` Bu senaryoda alt ve üst sınırı ters sırada girdiğiniz için, ` +
        `sistem bu değerleri küçükten büyüğe sıralayarak ${typedDesc} ` +
        `temsil eden bir aralık olarak değerlendirmiştir. ` +
        `Model, bu aralığın eğitimde gördüğü fiyat aralığı ile kesişimini alarak ` +
        `${tlFormat(data.effective_min)}–${tlFormat(
          data.effective_max
        )} TL arasında tarama yapmıştır.`;
    } else {
      rangeSentence =
        ` Siz ${typedDesc} seçtiniz. Model, bu aralığı eğitimde gördüğü fiyat aralığı ` +
        `ile kesiştirerek ${tlFormat(data.effective_min)}–${tlFormat(
          data.effective_max
        )} TL arasında fiyatları taramıştır.`;
    }
  } else {
    // Kullanıcı aralık girmedi
    rangeSentence =
      ` Siz herhangi bir fiyat aralığı belirtmediğiniz için, model ` +
      `eğitim aralığı içinde otomatik olarak ${tlFormat(
        data.effective_min
      )}–${tlFormat(
        data.effective_max
      )} TL aralığında arama yapmıştır.`;
  }

  const explanation = `Bu senaryoda model, rakip ortalama fiyatı ${tlFormat(
    data.competitor_price
  )} TL iken,${rangeSentence} Bu fiyatta beklenen talep ${Number(
    data.expected_qty
  ).toFixed(
    2
  )} adet, beklenen gelir ise yaklaşık ${tlFormat(
    data.expected_revenue
  )} TL’dir. Grafiklerde, farklı fiyat seviyelerinde talebin ve gelirin nasıl değiştiğini görebilirsiniz.`;

  document.getElementById("resScenarioText").textContent = explanation;

  resultsSection.style.display = "block";

  // 3.2 Grafikler
  const priceGrid = data.price_grid || [];
  const qtyGrid = data.qty_grid || [];
  const revGrid = data.revenue_grid || [];

  // Önce grafik bölümünü görünür yap (Chart.js doğru boyutu hesaplasın)
  chartsSection.style.display = "block";

  const demandCtx = document.getElementById("demandChart").getContext("2d");
  const revenueCtx = document.getElementById("revenueChart").getContext("2d");

  if (demandChart) demandChart.destroy();
  if (revenueChart) revenueChart.destroy();

  demandChart = new Chart(demandCtx, {
    type: "line",
    data: {
      labels: priceGrid.map((v) => tlFormat(v)),
      datasets: [
        {
          label: "Beklenen Talep (adet)",
          data: qtyGrid,
          fill: false,
          tension: 0.2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: {
        duration: 500,
        easing: "easeOutQuad",
      },
      scales: {
        x: {
          title: { display: true, text: "Fiyat (TL)" },
        },
        y: {
          title: { display: true, text: "Beklenen Talep (adet)" },
        },
      },
    },
  });

  revenueChart = new Chart(revenueCtx, {
    type: "line",
    data: {
      labels: priceGrid.map((v) => tlFormat(v)),
      datasets: [
        {
          label: "Beklenen Gelir (TL)",
          data: revGrid,
          fill: false,
          tension: 0.2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: {
        duration: 500,
        easing: "easeOutQuad",
      },
      scales: {
        x: {
          title: { display: true, text: "Fiyat (TL)" },
        },
        y: {
          title: { display: true, text: "Beklenen Gelir (TL)" },
        },
      },
    },
  });

  // Sonuçlara otomatik kaydır
  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}
