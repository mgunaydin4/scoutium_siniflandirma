######################################################
# Makine Öğrenmesi ile Yetenek Avcılığı Sınıflandırma
######################################################

######################################################
# İş Problemi

# Scout’lar tarafından izlenen futbolcuların özelliklerine verilen puanlara göre, oyuncuların hangi sınıf (average, highlighted) oyuncu olduğunu tahminleme.

######################################################

######################################################
#Veri Seti Hikayesi

# Veri seti Scoutium’dan maçlarda gözlemlenen futbolcuların özelliklerine göre scoutların değerlendirdikleri
# futbolcuların, maç içerisinde puanlanan özellikleri ve puanlarını içeren bilgilerden oluşmaktadır.
######################################################


# Adım 1: scoutium_attributes.csv ve scoutium_potential_labels.csv dosyalarını okutunuz.

# Gerekli kütüphanelerin yüklenmesi
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.model_selection import cross_validate
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)



att_data = pd.read_csv('Machine Learning 3. Hafta/Scoutium-220805-075951/scoutium_attributes.csv', sep=";")
potential_data = pd.read_csv('Machine Learning 3. Hafta/Scoutium-220805-075951/scoutium_potential_labels.csv', sep=";")

# Adım 2: Okutmuş olduğumuz csv dosyalarını merge fonksiyonunu kullanarak birleştiriniz.

# Verileri birleştir
df = pd.merge(att_data, potential_data,
                        on=['task_response_id', 'match_id', 'evaluator_id', 'player_id'],
                        how='inner')  # veya 'outer', 'left', 'right' ihtiyacınıza göre

df.head()
df.shape

# Adım 3: position_id içerisindeki Kaleci (1) sınıfını veri setinden kaldırınız.

df = df[df["position_id"] != 1]
df.position_id.value_counts()
df.shape
# Adım 4: potential_label içerisindeki below_average sınıfını veri setinden kaldırınız.( below_average sınıfı tüm verisetinin %1'ini oluşturur)

df = df[df["potential_label"] != "below_average"]
df.shape
df["attribute_id"].value_counts()
# Adım 5: Oluşturduğunuz veri setinden “pivot_table” fonksiyonunu kullanarak bir tablo oluşturunuz. Bu pivot table'da her satırda bir oyuncu olacak şekilde manipülasyon yapınız.

# Adım 5.1: indekste “player_id”,“position_id” ve “potential_label”, sütunlarda “attribute_id” ve değerlerde scout’ların oyunculara verdiği puan “attribute_value” olacak şekilde pivot table’ı oluşturunuz.
df.isnull().sum()
df = df.pivot_table(index= ["player_id", "position_id", "potential_label"], columns='attribute_id', values="attribute_value")
df.head()

# Adım 5.2: “reset_index” fonksiyonunu kullanarak indeksleri değişken olarak atayınız ve “attribute_id” sütunlarının isimlerini stringe çeviriniz.
df = df.reset_index()
df.head()
df.columns = df.columns.map(str)

df[df["4322"] > 80]

# Adım6: LabelEncoder fonksiyonunu kullanarak “potential_label” kategorilerini(average,highlighted)sayısal olarak ifade ediniz.
# Label Encoding işlemi
df["potential_label"].value_counts()
# average => 215
# highlihted => 56

le = LabelEncoder()
df['potential_label'] = le.fit_transform(df['potential_label'])
df["potential_label"].value_counts()
# 0 => 215
# 1 => 56

# Adım 7: Sayısal değişken kolonlarını “num_cols” adıyla bir listeye atayınız.
df.info()

df.head()
num_cols = [col for col in df.columns if "player_id" not in col and "potential_label" not in col]

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
df.head()

# Adım 9: Elimizdeki veri seti üzerinden minimum hata ile futbolcuların potansiyel etiketlerini tahmin eden bir makine öğrenmesi modeli geliştiriniz. (Roc_auc, f1, precision, recall, accuracy metriklerini yazdırınız.)

y = df["potential_label"]
X = df[num_cols]

log_model = LogisticRegression().fit(X, y)
log_model.intercept_
log_model.coef_

y_pred = log_model.predict(X)

# Model Evaluation
def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

plot_confusion_matrix(y, y_pred)

print(classification_report(y, y_pred))

# precision => 0.85
# recall => 0.61
# f1 => 0.71
# acc => 0.90

# ROC AUC
y_prob = log_model.predict_proba(X)[:, 1]
roc_auc_score(y, y_prob)

# Model Validation: Holdout
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20, random_state=17)

log_model = LogisticRegression().fit(X_train, y_train)

y_pred = log_model.predict(X_test)
y_prob = log_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))

# precision => 0.83
# recall => 0.36
# f1 => 0.50
# acc => 0.82

# ROC eğrisini hesapla
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# ROC eğrisini çiz
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='LogisticRegression (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()


# Model Validation: 10-Fold Cross Validation
y = df["potential_label"]
X = df[num_cols]

log_model = LogisticRegression().fit(X, y)

cv_results = cross_validate(log_model,
                            X, y,
                            cv=10,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])
# precision => 0.85
# recall => 0.61
# f1 => 0.71
# acc => 0.90

##################################

# precision => 0.83
# recall => 0.36
# f1 => 0.50
# acc => 0.82

cv_results['test_accuracy'].mean()
# Accuracy: 0.8525

cv_results['test_precision'].mean()
# Precision: 0.7738

cv_results['test_recall'].mean()
# Recall: 0.4905

cv_results['test_f1'].mean()
# F1-score: 0.5684

cv_results['test_roc_auc'].mean()
# AUC: 0.8460

# Prediction for A New Observation
X.columns

random_user = X.sample(1, random_state=45)
log_model.predict(random_user)


log_model.predict(X)
df["player_id"].shape

df.head()

last_df = pd.DataFrame({
    "player_id": df["player_id"],
    "real_value": df["potential_label"],
    "predict": log_model.predict(X)
})

last_df.head(20)