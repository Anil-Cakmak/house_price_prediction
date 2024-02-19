from helper import *
import missingno as msno
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
from pathlib import Path

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

#########################
# KEŞİFÇİ VERİ ANALİZİ
#########################
train = pd.read_csv("machine_learning/train.csv")
test = pd.read_csv("machine_learning/test.csv")
df_ = pd.concat([train, test], ignore_index=True)
df = df_.copy()

pd.set_option("display.max_rows", None)
check_df(df)  # (2919, 81)
pd.reset_option("display.max_rows", None)

# NA değerleri bazı özellikler için, eksik değerleri değil,
# ilgili değişkenin karşılık geldiği özelliğin olmaması durumu ifade ediyor.

bsmt_feats = df.loc[:, df.columns.str.contains("Bsmt")].columns  # no basement
garage_feats = df.loc[:, df.columns.str.contains("Garage")].columns  # no garage
pool_feats = df.loc[:, df.columns.str.contains("Pool")].columns  # no pool
fence_feats = df.loc[:, df.columns.str.contains("Fence")].columns  # no fence
misc_feats = df.loc[:, df.columns.str.contains("Misc")].columns  # no extra feature
fireplace_feats = df.loc[:, df.columns.str.contains("Fire")].columns  # no fireplace
mas_vnr_feats = df.loc[:, df.columns.str.contains("MasVnr")].columns  # no Masonry veneer
pd.set_option("display.max_row", None)
#############################################
df[bsmt_feats].isnull().sum()
print(df[(df.BsmtCond.isnull()) & (df.BsmtFinType1.notnull())][bsmt_feats])
df[(df["BsmtFinType1"].isnull())][bsmt_feats].isnull().sum()
df[(df["BsmtFinType1"].isnull())][bsmt_feats].sum()
# Anlıyoruz ki basement olmama durumunu en iyi "BsmtFinType1" değişkeni yansıtıyor.

df[garage_feats].isnull().sum()
print(df[(df.GarageFinish.isnull()) & (df.GarageType.notnull())][garage_feats])
df[(df["GarageType"].isnull())][garage_feats].isnull().sum()
df[(df["GarageType"].isnull())][garage_feats].sum()
# Anlıyoruz ki garage olmama durumunu en iyi "GarageType" değişkeni yansıtıyor.

df[pool_feats].isnull().sum()
len(df[(df.PoolQC.isnull()) & (df.PoolArea == 0)])
# Anlıyoruz ki havuz olmama durumunu en iyi "PoolQC" değişkeninin NA ve "PoolArea" değişkeninin 0 olması
# durumu ifade ediyor.

df[bsmt_feats].isnull().sum()
print(df[(df.MiscFeature.isnull()) & (df.MiscVal != 0)][misc_feats])
# Anlıyoruz ki ekstra özellik olmama durumunu en iyi "MiscFeature" değişkeninin NA ve "MiscVal" değişkeninin 0 olması
# durumu ifade ediyor.

df[fireplace_feats].isnull().sum()
print(len(df[(df.FireplaceQu.isnull()) & (df.Fireplaces == 0)]))
# Anlıyoruz ki fireplace olmama durumunu "FireplaceQu" değişkeni yansıtıyor.

df["MasVnrType"].isnull().sum()
df["MasVnrArea"].isnull().sum()
len(df[(df["MasVnrType"].isnull()) & (df["MasVnrArea"] == 0)])
# Anlıyoruz ki Masonry veneer olmama durumunu en iyi "MasVnrType" ve değişkeninin NA ve "MasVnrArea" değişkeninin 0
# olması durumu ifade ediyor.
pd.reset_option("display.max_row", None)

############################################################
# Numerik ve kategorik değişkenlerin yakalanması ve analizi.
############################################################
date_cols = ["YearBuilt", "YearRemodAdd", "GarageYrBlt", "YrSold", "MoSold"]

cat_cols, num_cols, cat_but_car = grab_col_names(df.drop(date_cols, axis=1), cat_th=20)
# Observations: 2919
# Variables: 76
# cat_cols: 55
# num_cols: 20
# cat_but_car: 1
# num_but_cat: 13

num_cols.remove("Id")

print(cat_but_car[0])  # Neighborhood
df.Neighborhood.nunique()  # 25
cat_summary(df, col_name="Neighborhood")

# Bir evin lokasyonunun fiyatını doğrudan etkileyeceğini düşündüğümden,bu değişkeni de kategorik değişkenlere ekliyorum.
cat_cols.append("Neighborhood")

for col in cat_cols:
    cat_summary(df, col)
    target_summary_with_cat(df, "SalePrice", col)

for col in num_cols:
    num_summary(df, col, plot=True)
    if col != "SalePrice":
        target_summary_with_num(df, "SalePrice", col)

##################################
# Aykırı ve eksik gözlemlerin incelenmesi.
##################################
for col in num_cols:
    if check_outlier(df, col):
        sns.boxplot(x=df[col])
        plt.show(block=True)

msno.heatmap(df[bsmt_feats])
plt.show(block=True)

#######################
# Korelasyon Analizi
#######################
df_corr(df[num_cols])
high_correlated_cols(df[num_cols], corr_th=0.5, plot=True)

corr = df[num_cols].corr().stack().drop_duplicates().sort_values(ascending=False)
print(corr[lambda x: x > 0.5])

######################
# FEATURE ENGINEERING
######################

# Eksik gözlemlerin halledilmesi.
bsmt_objs = [col for col in bsmt_feats if df[col].dtypes == "O"]
bsmt_nums = [col for col in bsmt_feats if col not in bsmt_objs]
df[bsmt_feats].isnull().sum()

for i in range(len(df)):
    if pd.isnull(df.at[i, "BsmtFinType1"]):
        for col in bsmt_objs:
            df.loc[i, col] = "no_basement"
        for col in bsmt_nums:
            df.loc[i, col] = 0

garage_objs = [col for col in garage_feats if df[col].dtypes == "O"]
garage_nums = [col for col in garage_feats if col not in garage_objs]
garage_nums.remove("GarageYrBlt")
df[garage_feats].isnull().sum()
for i in range(len(df)):
    if pd.isnull(df.at[i, "GarageType"]):
        for col in garage_objs:
            df.loc[i, col] = "no_garage"
        for col in garage_nums:
            df.loc[i, col] = 0

for i in range(len(df)):
    if pd.isnull(df.at[i, "PoolQC"]) and df.loc[i, "PoolArea"] == 0:
        df.loc[i, "PoolQC"] = "no_pool"

print(df[(df["MiscFeature"].notnull()) & (df["MiscVal"] == 0)][misc_feats])
# "MiscFeature" NA değeri almadığında "MiscVal"ın 0 değeri alması eksik gözlem durumunu ifade ediyor olabilir.
# Ek olarak bu 0 değerleri NA olarak değiştiriliyor.
for i in range(len(df)):
    if pd.isnull(df.at[i, "MiscFeature"]) and df.loc[i, "MiscVal"] == 0:
        df.loc[i, "MiscFeature"] = "no_misc"
    elif pd.notnull(df.at[i, "MiscFeature"]) and df.loc[i, "MiscVal"] == 0:
        df.loc[i, "MiscVal"] = np.nan

print(df[(df["MasVnrType"].notnull()) & (df["MasVnrArea"] == 0)][mas_vnr_feats])
# "MasVnrType" na değeri almadığında "MasVnrArea"nın 0 değeri alması eksik gözlem durumunu ifade ediyor olabilir.
# Ek olarak bu 0 değerleri na olarak değiştiriliyor.
for i in range(len(df)):
    if pd.isnull(df.at[i, "MasVnrType"]) and ((df.loc[i, "MasVnrArea"] == 0) | pd.isnull(df.at[i, "MasVnrArea"])):
        df.loc[i, "MasVnrType"] = "no_mas_vnr"
    elif pd.notnull(df.at[i, "MasVnrType"]) and df.loc[i, "MasVnrArea"] == 0:
        df.loc[i, "MasVnrArea"] = np.nan

df["Fence"].fillna("no_fence", inplace=True)
df["FireplaceQu"].fillna("no_fireplace", inplace=True)
df["Alley"].fillna("no_alley_access", inplace=True)

# Yeni türettiğim yaş değişkenlerinde negatif olmama durumunu gözlemlediğimden bu gözlemlere NA değeri atayıp
# sonrasında bu yaş değişkenleri diğer yıl değişkenleriyle birlikte KNN modeline alarak bu eksik değerleri
# tahmin ediyorum. (NOT: Hiperparametre optimizasyonları yapılıp model kurulmadan önce gözden kaçması sonucu 0 olan
# değişkenlere de NA değeri verilmişti.)
df["new_building_age"] = df.YrSold - df.YearBuilt
df["new_remodel_age"] = df.YrSold - df.YearRemodAdd
df["new_building_age"] = df["new_building_age"].apply(lambda x: x if x >= 0 else np.nan)
df["new_remodel_age"] = df["new_remodel_age"].apply(lambda x: x if x >= 0 else np.nan)
df["new_is_summer"] = df.MoSold.apply(lambda x: 1 if x in [5, 6, 7, 8, 9] else 0)

knn_cols = ["YearBuilt", "YearRemodAdd", "YrSold", "new_building_age", "new_remodel_age"]
scaler = MinMaxScaler()
df[knn_cols] = pd.DataFrame(scaler.fit_transform(df[knn_cols]), columns=knn_cols)
imputer = KNNImputer(n_neighbors=5)
df[knn_cols] = pd.DataFrame(imputer.fit_transform(df[knn_cols]), columns=knn_cols)
df[knn_cols] = pd.DataFrame(scaler.inverse_transform(df[knn_cols]), columns=knn_cols)

# Pek çok eksik değer barındıran LotFrontage ve GarageYrBlt değişkenleri tahmin başarısına yeterli katkıyı sağlayacağı
# düşünülmediğinden kullanılmadan veri setinden çıkarılıyor.
df.drop(["YearBuilt", "YearRemodAdd", "GarageYrBlt", "YrSold", "LotFrontage"], axis=1, inplace=True)

df = quick_missing_imp(df, target="SalePrice")

# Ordinal Encoding
df[["BsmtQual", "BsmtCond"]] = df[["BsmtQual", "BsmtCond"]] \
    .replace({"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "no_basement": 0})
df[["GarageQual", "GarageCond"]] = df[["GarageQual", "GarageCond"]] \
    .replace({"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "no_garage": 0})
df[["KitchenQual", "ExterQual", "ExterCond", "HeatingQC"]] = \
    df[["KitchenQual", "ExterQual", "ExterCond", "HeatingQC"]].replace({"Ex": 4, "Gd": 3, "TA": 2, "Fa": 1, "Po": 0})
df["FireplaceQu"] = df["FireplaceQu"].replace({"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "no_fireplace": 0})
df["BsmtExposure"] = df["BsmtExposure"].replace({"Gd": 3, "Av": 2, "Mn": 1, "No": 0, "no_basement": 0})
df["GarageFinish"] = df["GarageFinish"].replace({"Fin": 3, "RFn": 2, "Unf": 1, "no_garage": 0})
df["PoolQC"] = df["PoolQC"].replace({"Ex": 4, "Gd": 3, "TA": 2, "Fa": 1, "no_pool": 0})
df["Functional"] = df["Functional"].replace({"Typ": 7, "Min1": 6, "Min2": 5, "Mod": 4,
                                             "Maj1": 3, "Maj2": 2, "Sev": 1, "Sal": 0})
df[["BsmtFinType1", "BsmtFinType2"]] = df[["BsmtFinType1", "BsmtFinType2"]].\
    replace({"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "no_basement": 0})

df["LandSlope"] = df["LandSlope"].replace({"Gtl": 3, "Mod": 2, "Sev": 1})

# "PoolArea" değişkeni için değerlerin yüzde 99'dan fazlası, havuz olmaması durumundan kaynaklı olarak,
# 0 değerini alıyor. Bunun için direkt olarak rare encoding uyguluyoruz.
df["PoolArea"] = df["PoolArea"].apply(lambda x: 1 if x > 0 else x)

ordinal_cols = ["BsmtQual", "BsmtCond", "GarageQual", "GarageCond", "KitchenQual", "FireplaceQu", "BsmtExposure",
                "GarageFinish", "PoolQC", "ExterQual", "ExterCond", "HeatingQC", "Functional",
                "BsmtFinType1", "BsmtFinType2"]

no_encode_cols = ["OverallQual", "OverallCond", "BsmtFullBath", "BsmtHalfBath", "FullBath", "BedroomAbvGr",
                  "KitchenAbvGr", "Fireplaces", "GarageCars", "TotRmsAbvGrd", "HalfBath"]


# Yeni Feature'lar türetilmesi.

df["new_total_qual"] = df[[*ordinal_cols, "OverallQual", "OverallCond"]].sum(axis=1)

# Total Floor
df["new_totalflrsF"] = df["1stFlrSF"] + df["2ndFlrSF"]

# Total Finished Basement Area
df["new_totalbsmtfinsf"] = df.BsmtFinSF1 + df.BsmtFinSF2

df["new_livratio"] = df.GrLivArea / df.LotArea

# Total Porch Area
df["new_total_porcharea"] = df.OpenPorchSF + df.EnclosedPorch + df.ScreenPorch + df["3SsnPorch"] + df.WoodDeckSF

df["new_masvnrratio"] = df["MasVnrArea"] / df["GrLivArea"]

df["new_bsmt_baths"] = df["BsmtFullBath"] + df["BsmtHalfBath"]
df["new_baths_above"] = df["FullBath"] + df["HalfBath"]
df["new_other_rooms"] = df["TotRmsAbvGrd"] - df["BedroomAbvGr"]

cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=13)
# Observations: 2919
# Variables: 88
# cat_cols: 58
# num_cols: 29
# cat_but_car: 1
# num_but_cat: 32

cat_cols.append("Neighborhood")
num_cols.remove("Id")
num_cols.remove("SalePrice")

# Nümerik değişkenlerin ölçeklendirilmesi.
X_scaled = RobustScaler().fit_transform(df[num_cols])
df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)

# Rare encoding.
rare_cols = [col for col in cat_cols if col not in [*ordinal_cols, *no_encode_cols]]
rare_analyser(df, "SalePrice", rare_cols)
df[rare_cols] = rare_encoder(df[rare_cols], rare_perc=0.01)
df = one_hot_encoder(df, rare_cols)
check_df(df)

df.columns = df.columns.str.replace(" ", "_")

# Model için oluşturulan veri setinin kaydedilmesi.
filepath = Path("house_price_prediction/df.csv")
filepath.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(filepath, index=False)


