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
