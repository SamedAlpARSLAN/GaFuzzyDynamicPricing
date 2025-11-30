Bu projede Kaggle’daki Retail Price Optimization (retail_price.csv) veri seti kullanılarak
Genetik Algoritma ve Bulanık Mantık tabanlı bir dinamik fiyatlandırma modeli tasarlandı.
Çalışma, “GA ve Fuzzy Logic ile Dinamik Fiyatlandırma Modeli” başlıklı final projesi kapsamında yapılmıştır.

Veri setinde bir e-ticaret sitesine ait ürünlerin aylık satış miktarı (qty), birim fiyatı (unit_price) ve
üç farklı rakibin fiyatları (comp_1, comp_2, comp_3) bulunmaktadır.

Modelde girişler, ürünün kendi fiyatı ve rakiplerin ortalama fiyatıdır. Bu iki değişken için
üçgen üyelik fonksiyonları ile düşük / orta / yüksek terimleri tanımlanmıştır.

Böylece “fiyat” × “rakip fiyatı” kombinasyonlarından oluşan 9 adet bulanık kural kurulmuştur.
Çıkış değişkeni, ürünün beklenen talebidir (qty).

Her kuralın “THEN qty ≈ k” sabiti, Genetik Algoritma ile öğrenilmektedir. GA’nın fitness fonksiyonu,
gerçek talep ile bulanık modelin tahmini talebi arasındaki ortalama karesel hatanın (MSE) negatifidir.

Eğitilmiş fuzzy+GA modeli kullanılarak, belirli bir rakip fiyatı için farklı fiyat adayları denenmiş ve
beklenen gelir = fiyat × talep hesabı yapılarak geliri maksimize eden dinamik fiyat seçilmiştir.

Böylece literatürdeki “fuzzy kural tabanlı talep tahmini ile dinamik fiyatlandırma” yaklaşımı basit bir şekilde
gerçek bir veri seti üzerinde uygulanmış, fiyat-tarafa duyarlı talep fonksiyonu GA ile optimize edilmiştir.