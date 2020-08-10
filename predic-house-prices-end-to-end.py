# Python ≥3.5 is required
# Scikit-Learn ≥0.20 is required
# Common imports

# To plot pretty figures
#%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

import os

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    # savefig() method is used to save the figure created after plotting data.
    plt.savefig(path, format=fig_extension, dpi=resolution)

# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")


#Get the data

import os
import tarfile
import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
#قم بفصل المسار باستخدام الرمز (‘/’)
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    # اذا كان المسار غير موجود قم بانشاء المجلدات
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    #  استرجاع مورد عبر الرابط وتخزينه في موقع مؤقت وهو المسار المحدد
    urllib.request.urlretrieve(housing_url, tgz_path)
    #فتح الملف كملف مضغوط
    housing_tgz = tarfile.open(tgz_path)
    #فك ضغط الملف في المسار المحدد
    housing_tgz.extractall(path=housing_path)
    # اغلاق الملف
    housing_tgz.close()

#استدعاء الدالة لقراءة البيانات
fetch_housing_data()

#  CSV قراء ملف
import pandas as pd

# قم بقراءة البيانات الموجودة في ملف مفصولة القيم فيه بفواصل (CSV) من خلال دالة read_csv من مكتبة الباندا.
#تُرجع هذه الدالة  DataFrame object للباندا يحتوي على جميع البيانات
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

#استدعاء الدالة لقراءة البيانات المخزنة
housing = load_housing_data()

#طباعة الصفوف الخمسة الأولى من DataFrame
housing.head()

# معلومات الداتا ست housing
housing.info()

#تستخدم للحصول على سلسلة تحتوي على عدد القيم الغير متكررة.
housing["ocean_proximity"].value_counts()

#دعونا نلقي نظرة على الحقول الأخرى من خلال دالة describe والتي تظهر ملخص الصفات (السمات) العددية.
housing.describe()

#رسم البيانات بيانياً على شكل أعمدة بحيث أن هناك عمود لكل سمة عددية
#matplotlib inline
import matplotlib.pyplot as plt
#bins   :	Number of histogram bins to be used.
#figsize: tuple, The size in inches of the figure to create.
housing.hist(bins=50, figsize=(20,15))
#defined in the setup above
save_fig("attribute_histogram_plots")
plt.show()



import numpy as np

#تثبيت بذرة مولد الأرقام العشوائية 
np.random.seed(42)

# For illustration only. Sklearn has train_test_split()
def split_train_test(data, test_ratio):
    #Generate a random permutation of elements
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    #The iloc indexer for Pandas Dataframe is used for integer-location based indexing / selection by position.
    # It returns Pandas DataFrame when multiple rows are selected
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing, 0.2)
len(train_set)

len(test_set)

from zlib import crc32

def test_set_check(identifier, test_ratio):
    #تقوم بتوليد نفس القيمة الرقمية عبر جميع إصدارات وأنظمة Python
    # ** is power
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    #lambda anonymous function
    #يطبق دالة test_set_check الخاصة بنا على المعرفات الفريدة لكل عينة
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    #loc Access a group of rows and columns by label(s) or a boolean array.
    #The ~ flips 1s to 0s and 0s to 1s.
    return data.loc[~in_test_set], data.loc[in_test_set]

housing_with_id = housing.reset_index()   # adds an `index` column
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

#خط الطول وخط العرض لمنطقة معينة هي قيم ثابتة دائماً، لذا يمكنك دمجها كمعرف كالتالي:
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")

test_set.head()

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

test_set.head()

housing["median_income"].hist()

#دالة pd.cut لإنشاء سمة فئة دخل بخمس فئات (من 1 إلى 5)
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

housing["income_cat"].value_counts()

#يتم تمثيل فئات الدخل هذه من خلال الشكل التالي
housing["income_cat"].hist()

# أخذ عينات طبقية بناءً على فئة الدخل
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42) # split is object
# Method: split(X, y[, groups]), Generate indices to split data into training and test set.
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

strat_test_set["income_cat"].value_counts() / len(strat_test_set)
