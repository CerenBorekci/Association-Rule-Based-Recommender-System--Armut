                                    # İş Problemi

"""
 Türkiye’nin en büyük online hizmet platformu olan Armut, hizmet verenler ile hizmet almak isteyenleri buluşturmaktadır.
Bilgisayarın veya akıllı telefonunun üzerinden birkaç dokunuşla temizlik, tadilat, nakliyat gibi hizmetlere kolayca
ulaşılmasını sağlamaktadır.
Hizmet alan kullanıcıları ve bu kullanıcıların almış oldukları servis ve kategorileri içeren veri setini kullanarak
Association Rule Learning ile ürün tavsiye sistemi oluşturulmak istenmektedir.

"""

                                     # Veri Seti
"""
Veri seti müşterilerin aldıkları servislerden ve bu servislerin kategorilerinden oluşmaktadır.
Alınan her hizmetin tarih ve saat bilgisini içermektedir.
# UserId: Müşteri numarası
# ServiceId: Her kategoriye ait anonimleştirilmiş servislerdir. (Örnek : Temizlik kategorisi altında koltuk yıkama servisi)
# Bir ServiceId farklı kategoriler altında bulanabilir ve farklı kategoriler altında farklı servisleri ifade eder.
# (Örnek: CategoryId’si 7 ServiceId’si 4 olan hizmet petek temizliği iken CategoryId’si 2 ServiceId’si 4 olan hizmet mobilya montaj)
# CategoryId: Anonimleştirilmiş kategorilerdir. (Örnek : Temizlik, nakliyat, tadilat kategorisi)
# CreateDate: Hizmetin satın alındığı tarih
"""



                                # GÖREV 1: Veriyi Hazırlama
#Öncelikle kütüphanelerimizi import edelim ve veri setimizi çağıralım.

import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules
import datetime as dt

df_ = pd.read_csv(r"HAFTA 5/CASE STUDY 1/armut_data.csv")
df = df_.copy()
df.head()
df.dtypes
df.shape #(162523, 4)
df.isnull().sum() #hepsi dolu nan yok
df.value_counts()
df.dropna(inplace=True)
df.describe().T


# Adım 2: ServisID her bir CategoryID özelinde farklı bir hizmeti temsil etmektedir.
# ServiceID ve CategoryID'yi "_" ile birleştirerek hizmetleri temsil edecek yeni bir değişken oluşturunuz.
df.head()
#Burada ServiceId ve CategoryId değişkenlerini gözlemlediğimde 1. ve 2. sütunda olduğunu buldum. Sütuna göre birleştirme
#işlemi gerçekleştireceğim.

       #1. Yöntem
df["Hizmet"] = [str(row[1]) + "_" + str(row[2]) for row in df.values]
df.head()


       #2. Yöntem
df["ServiceId_1"] = df["ServiceId"].astype(str)
df["CategoryId_1"] = df["CategoryId"].astype(str)

df["Hizmet"] = ["_".join(i) for i in (df[["ServiceId_1","CategoryId_1"]].values)]

#Böylece Hizmet adında yeni bir değişken atamış olduk.



# Adım 3: Veri seti hizmetlerin alındığı tarih ve saatten oluşmaktadır, herhangi bir sepet tanımı (fatura vb. ) bulunmamaktadır.
"""
#Association Rule Learning uygulayabilmek için bir sepet (fatura vb.) tanımı oluşturulması gerekmektedir.
Burada sepet tanımı her bir müşterinin aylık aldığı hizmetlerdir.
Örneğin; 7256 id'li müşteri 2017'in 8.ayında aldığı 9_4, 46_4 hizmetleri bir sepeti;
2017’in 10.ayında aldığı  9_4, 38_4  hizmetleri başka bir sepeti ifade etmektedir. Sepetleri unique bir ID ile
tanımlanması gerekmektedir.Bunun için öncelikle sadece yıl ve ay içeren yeni bir date değişkeni oluşturunuz.
UserID ve yeni oluşturduğunuz date değişkenini kullanıcı bazında "_" ile birleştirirek ID adında yeni bir değişkene 
atayınız.
"""
#Öncelikle değişkenimiz CreateDate için yeni bir dataframe oluşturalım ve yeni değişken oluşturarak bu dataframei yıl ve ay
#bazında bölüp değer atayalım.

df["CreateDate"] = pd.to_datetime(df["CreateDate"])
df["New_Date"] = df["CreateDate"].dt.strftime("%Y-%m")
df["UserId"]=df["UserId"].astype(str)
df["New_Date"]=df["New_Date"].astype(str)

df["SepetID"] = ["_".join(i) for i in df[["UserId","New_Date"]].values]
df["SepetID"].head()
#SepetID değişkenimizde UserID bilgilerimiz ve CreateDate değişkeninin yıl ve ay bazında kırılımından oluşan yeni değerler
#bulunmaktadır.

                                # GÖREV 2: Birliktelik Kuralları Üretiniz
# Adım 1: Aşağıdaki gibi sepet hizmet pivot table’i oluşturunuz.

"""
Hizmet         0_8  10_9  11_11  12_7  13_11  14_7  15_1  16_8  17_5  18_4..
SepetID
0_2017-08        0     0      0     0      0     0     0     0     0     0..
0_2017-09        0     0      0     0      0     0     0     0     0     0..
0_2018-01        0     0      0     0      0     0     0     0     0     0..
0_2018-04        0     0      0     0      0     1     0     0     0     0..
10000_2017-08    0     0      0     0      0     0     0     0     0     0..

"""


               #1. Yöntem
new_product= df.pivot_table(index=["SepetID"], columns=["Hizmet"], aggfunc={"Hizmet": "count"}).fillna(0). \
        applymap(lambda x: 1 if x != 0 else 0)
new_product.head()

#Eğer reset_index yaparsam hesaplamada hatalı sonuç verebilir ve SepetID değişkenimin değerleri kayabilir.
               #2. Yöntem
new_product= df.groupby(['SepetID', 'Hizmet'])["Hizmet"]. \
    count(). \
    unstack(). \
    fillna(0). \
    applymap(lambda x: 1 if x > 0 else 0)
new_product.head()
df_t=new_product
df_t.head()


# Adım 2: Birliktelik kurallarını oluşturunuz.
frequent_itemsets = apriori(new_product,
                            min_support=0.01,
                            use_colnames=True)

frequent_itemsets.sort_values("support",ascending=False)

rules = association_rules(frequent_itemsets,
                          metric="support",
                          min_threshold=0.01)

#Burada bir sınıflandırma yaptık. Support değeri, confidence ve lift değeri 0.1den daha bütük olan dataframei
# confidence değişkenine göre azalan şekilde sıraladık.
rules[(rules["support"]>0.01)&(rules["confidence"]>0.1)&(rules["lift"]>1)].sort_values("confidence",ascending=False)

#Lift değişkenine göre azalan şekilde sıraladık.
sorted_rules = rules.sort_values("lift",ascending=False)
sorted_rules



#Adım 3: arl_recommender fonksiyonunu kullanarak en son 2_0 hizmetini alan bir kullanıcıya hizmet önerisinde bulununuz.
def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]

arl_recommender(rules, "2_0", 1)
arl_recommender(rules, "2_0", 2)