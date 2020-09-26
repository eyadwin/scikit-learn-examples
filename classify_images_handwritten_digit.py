# تصنيف صور أرقام مكتوبة بخط اليد


# تجهيز البرنامج
#########################

# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
#%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "classification"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
#########################3
#جلب صور الارقام MNIST
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1, cache=True)
mnist.keys()

#•	مؤشر DESCR يصف مجموعة البيانات.
#•	مؤشر data يحتوي على مصفوفة بها صف واحد لكل مثيل instance وعمود واحد لكل خاصية feature.
#•	مؤشر target يحتوي على مصفوفة التسميات labels.

X, y = mnist["data"], mnist["target"]
X.shape

y.shape

#%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt

some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap=mpl.cm.binary)
plt.axis("off")

save_fig("some_digit_plot")
plt.show()

y[0]

y = y.astype(np.uint8)

#تقسيم مجموعة بيانات MNIST إلى مجموعة تدريب (أول 60000 صورة) ومجموعة اختبار (آخر 10000 صورة)
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

#سيكون هذا مثال لمصنف ثنائي قادر على التمييز بين فئتين فقط الرقم هو 5 أو الرقم ليس 5.
# فلنقم بإنشاء متجهات النتيجة target vectors لـمهمة التصنيف هذه
y_train_5 = (y_train == 5) # True for all 5s, False for all other digits
y_test_5 = (y_test == 5)

#الآن لنختار مصنفًا ونقوم بتدريبه.
# أفضل مكان للبدء هو مصنف التدرج العشوائي Stochastic Gradient Descent (SGD) ، باستخدام كلاس SGDClassifier من مكتبةـ Scikit-Learn
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
sgd_clf.fit(X_train, y_train_5)

#الآن يمكننا استخدامه للكشف عن صور الرقم 5:
sgd_clf.predict([some_digit])


#في بعض الأحيان ، ستحتاج إلى مزيد من التحكم في عملية التحقق المتقاطع أكثر مما يوفره Scikit-Learn بشكل جاهز. في هذه الحالة يمكنك تنفيذ التحقق المتقاطع بنفسك
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3, random_state=42)

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))

#دعنا نستخدم الدالة cross_val_score  لتقييم نموذج SGDClassifier باستخدام التحقق المتقاطع K-fold بثلاث طيات folds

from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")

#دعونا نلقي نظرة على المصنف البسيط الذي يصنف فقط كل صورة في فئة ليس الرقم 5
from sklearn.base import BaseEstimator
class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)


never_5_clf = Never5Classifier()
cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")


from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

from sklearn.metrics import confusion_matrix

confusion_matrix(y_train_5, y_train_pred)

#أنت الآن جاهز للحصول على مصفوفة الالتباس باستخدام دالة confusion_matrix  فقط قم بتمرير الفئات المستهدفة (y_train_5)  والفئات المتوقعة (y_train_pred)
y_train_perfect_predictions = y_train_5  # pretend we reached perfection
confusion_matrix(y_train_5, y_train_perfect_predictions)


from sklearn.metrics import precision_score, recall_score

precision_score(y_train_5, y_train_pred)  # == 4096 / (4096 + 1522)
recall_score(y_train_5, y_train_pred) # == 4096 / (4096 + 1325)


from sklearn.metrics import f1_score

f1_score(y_train_5, y_train_pred)


y_scores = sgd_clf.decision_function([some_digit])
y_scores

threshold = 0
y_some_digit_pred = (y_scores > threshold)

y_some_digit_pred


#يستخدم مصنف SGDClassifier حدًا يساوي 0 ، لذا فإن الكود السابق يعيد نفس النتيجة لدالة predict وهي True. لنرفع الحدّ threshold
threshold = 8000
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method="decision_function")


from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.legend(loc="center right", fontsize=16) # Not shown in the book
    plt.xlabel("Threshold", fontsize=16)        # Not shown
    plt.grid(True)                              # Not shown
    plt.axis([-50000, 50000, 0, 1])             # Not shown



recall_90_precision = recalls[np.argmax(precisions >= 0.90)]
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]


plt.figure(figsize=(8, 4))                                                                  # Not shown
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.plot([threshold_90_precision, threshold_90_precision], [0., 0.9], "r:")                 # Not shown
plt.plot([-50000, threshold_90_precision], [0.9, 0.9], "r:")                                # Not shown
plt.plot([-50000, threshold_90_precision], [recall_90_precision, recall_90_precision], "r:")# Not shown
plt.plot([threshold_90_precision], [0.9], "ro")                                             # Not shown
plt.plot([threshold_90_precision], [recall_90_precision], "ro")                             # Not shown
save_fig("precision_recall_vs_threshold_plot")                                              # Not shown
plt.show()
