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

#النظر إلى فئة نسب الدخل في مجموعة الاختبار
strat_test_set["income_cat"].value_counts() / len(strat_test_set)

#مقارنة بين نسب فئة الدخل في مجموعة البيانات كاملة في مجموعة الاختبار التي تم إنشاؤها باستخدام أخذ العينات الطبقية stratified sampling ، وفي مجموعة اختبار تم إنشاؤها باستخدام أخذ العينات العشوائية random sampling.
def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

#dataframe.sort_index() function sorts objects by labels along the given axis,
# and returns the original DataFrame sorted by the labels.
#The default value is 0 which identifies the sorting by rows
compare_props = pd.DataFrame({
    "Overall": income_cat_proportions(housing),
    "Stratified": income_cat_proportions(strat_test_set),
    "Random": income_cat_proportions(test_set),
}).sort_index()
compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100

compare_props

#الآن علينا إزالة سمة الدخل حتى تعود البيانات إلى حالتها الأصلية:
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

#
#اكتشف وتصور البيانات لفهمها
#

#ننشئ نسخة حتى تتمكن من معالجتها دون الإضرار بمجموعة التدريب
housing = strat_train_set.copy()

#إنشاء مخطط مبعثر scatterplot  لجميع المناطق
housing.plot(kind="scatter", x="longitude", y="latitude")
save_fig("bad_visualization_plot")

#ضبط خيار ألفا إلى 0.1 لتسهيل تصور الأماكن ذات الكثافة العالية من نقاط البيانات (يعني بها منازل أكثر)
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
save_fig("better_visualization_plot")

# s:The size of each point.
#c: The color of each point. The column name values will be used to color the marker points according to a colormap
#cmap=plt.get_cmap("jet"), get colormap
#sharex=True, when enabling it takes the x-axis of the last subplot as the axis of the entire plot

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=housing["population"]/100, label="population", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False)
plt.legend()
save_fig("housing_prices_scatterplot")

#حساب معامل الارتباط القياسي (يسمى أيضًا Pearson's r) بين كل زوج من السمات
corr_matrix = housing.corr()

corr_matrix["median_house_value"].sort_values(ascending=False)

# from pandas.tools.plotting import scatter_matrix # For older versions of Pandas
from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
save_fig("scatter_matrix_plot")

#يظهر من الرسوم البيانية أن السمة الأفضل للتنبؤ بمتوسط قيمة المنزل median house value هي الوسيط
#للدخل  median income، لذلك سنقوم بتكبير الرسم البياني للارتباط correlation  لها

housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1)
plt.axis([0, 16, 0, 550000])
save_fig("income_vs_house_value_scatterplot")

#ننشئ هذه السمات الجديدة
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]

#والآن لنلقي نظرة على مصفوفة الارتباط مرة أخرى
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

#دعونا أولاً نعود إلى مجموعة تدريب نظيفة (بنسخ strat_train_set مرة أخرى)
housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set
housing_labels = strat_train_set["median_house_value"].copy()

sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
sample_incomplete_rows

#تخلص من المناطق التي لديها قيم مفقودة.
sample_incomplete_rows.dropna(subset=["total_bedrooms"])    # option 1

#تخلص من سمة total_bedrooms ككل
sample_incomplete_rows.drop("total_bedrooms", axis=1)       # option 2

#عيّن القيم إلى الأسطر المفقودة من السمة (مثلاً صفر ، متوسط القيم، الوسيط ، إلخ).
median = housing["total_bedrooms"].median()
sample_incomplete_rows["total_bedrooms"].fillna(median, inplace=True) # option 3

#سيكيت  ليرن class مفيد لمعالجة القيم المفقودة
#مع تحديد أنك تريد استبدال القيم المفقودة لكل سمة بمتوسط تلك السمة
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")

#نظرًا لأنه لا يمكن حساب الوسيط إلا على سمات عددية ، فأنت بحاجة إلى إنشاء ملف بنسخة من البيانات بدون السمة النصية ocean_proximity
housing_num = housing.drop("ocean_proximity", axis=1)

#يمكنك الآن ملاءمة object نسخة imputer مع بيانات التدريب باستخدام طريقة fit
imputer.fit(housing_num)

#قام imputer بحساب متوسط كل سمة وتخزين النتيجة في متغير statistics_  الخاص به.
imputer.statistics_

housing_num.median().values

#الآن يمكنك استخدام imputer المدرب لتحويل مجموعة التدريب عن طريق إستبدال القيم المفقودة مع المتوسطات التي تم حسابها:
X = imputer.transform(housing_num)

#والنتيجة هي مصفوفة NumPy بسيطة تحتوي على الميزات (السمات) المحولة. أذا أردت تحويلها مرة أخرى الى باندا DataFrame نقوم بالتالي:
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing.index)

housing_tr.loc[sample_incomplete_rows.index.values]

#الآن دعنا نعالج السمة الفئوية ocean_proximity:
housing_cat = housing[["ocean_proximity"]]
housing_cat.head(10)

# سنحوّل هذه الفئات من نص إلى أرقام. لهذا ، يمكننا استخدام class كلاس OrdinalEncoder من مكتبة Scikit-Learn
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat) #Fit to data, then transform it.
housing_cat_encoded[:10]

#قائمة الفئات categories  باستخدام المتغير العام  _categories
ordinal_encoder.categories_

#تحويل القيم الفئوية إلى متجه  one-hot
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot

#تحويلها إلى مصفوفة NumPy من خلال استدعاء الدالة toarray
housing_cat_1hot.toarray()

# يمكنك الحصول على قائمة الفئات باستخدام المتغير العام categories_:
cat_encoder.categories_

#محول transformer  صغير قمنا بتعريفه small transformer class تقوم باضافة سمات يتم دمجها  combined attributes
from sklearn.base import BaseEstimator, TransformerMixin

# column index
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

#BaseEstimator and TransformerMixin as base classes.
# The former one gives us get_params() and set_params() methods
# and the latter gives us fit_transform() method
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

housing_extra_attribs = pd.DataFrame(
    housing_extra_attribs,
    columns=list(housing.columns)+["rooms_per_household", "population_per_household"],
    index=housing.index)
housing_extra_attribs.head()


#سنستخدم Scikit-Learn كلاس Pipeline class  للمساعدة في تسلسلات التحولات. سنقوم هنا بسلسلة تحويلات صغيرة للسمات العددية
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)

housing_num_tr

#محوّل  transformer  واحد قادر على التعامل مع جميع الأعمدة ، يطبق التحولات المناسبة على كل عمود نستخدمه لتطبيق جميع التحولات لبيانات السكن
from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)

housing_prepared

# فلنبدأ أولاً بتدريب نموذج الانحدار الخطي
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

#لديك الآن نموذج انحدار خطي جاهز للعمل. فلنجربه على بعض الأمثلة من مجموعة التدريب:
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

print("Predictions:", lin_reg.predict(some_data_prepared))

print("Labels:", list(some_labels))

#إنه يعمل ، على الرغم من أن التنبؤات ليست دقيقة تمامًا. دعونا نقيس
# Root Mean Square Error (RMSE) لنموذج الانحدار هذا على التدريب بأكمله باستخدام دالة mean_squared_error لـ Scikit-Learn:
from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse

from sklearn.metrics import mean_absolute_error

lin_mae = mean_absolute_error(housing_labels, housing_predictions)
lin_mae

#لنجرب نموذجًا أكثر تعقيدًا لنرى كيف يعمل.
#دعونا ندرب DecisionTreeRegressor. هذا نموذج قوي قادر على إيجاد العلاقات غير الخطية المعقدة في البيانات
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)

#الآن وقد تم تدريب النموذج ، فلنقم بتقييمه على مجموعة التدريب:
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse

#تدريب نموذج شجرة القرار وتقييمه 10 مرات
# ، واختيار جزء مختلف للتقييم في كل مرة والتدريب على 9 أجزاء الأخرى. والنتيجة هي مصفوفة تحتوي على درجات التقييم العشر
from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

#فلننظر للنتائج
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)


#دعنا نحسب نفس الدرجات لنموذج الانحدار الخطي لمقارنة النتيجة مع شجرة القرار
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)

#لنجرب نموذجًا أخيرًا: RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(housing_prepared, housing_labels)

#بامكانك استخدام GridSearchCV من Scikit-Learn للبحث بدلاً منك.
# كل ما تحتاجه هو إخبارها المعاملات الفائقة التي تريدها أن تجربها وما هي القيم للتجربة
# ، وسيستخدم التحقق المتبادل cross-validation لتقييم جميع التوليفات الممكنة من قيم المعاملات الفائقة
from sklearn.model_selection import GridSearchCV

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)


#سيستكشف grid search بحث الشبكة 12 + 6 = 18 مجموعة من قيم  المعاملات الفائقة ل  RandomForestRegressor ،
# وسوف يتم تدريب كل نموذج 5 مرات> عندما يتم ذلك يمكنك الحصول على أفضل تركيبة من المعاملات كالتالي:
grid_search.best_params_

#يمكنك أيضًا الحصول على أفضل مقدّر مباشرةً:
grid_search.best_estimator_

#لنلقِ نظرة على درجة كل مجموعة من المعاملات الفائقة hyperparameter  والتي تم اختبارها أثناء البحث grid search
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances

#دعنا نعرض هذه الدرجات المهمة بجانب أسماء السمات المقابلة لها:

extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
#cat_encoder = cat_pipeline.named_steps["cat_encoder"] # old solution
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)

#احصل على المتنبئين predictors  والتسميات labels من مجموعة الاختبار ، قم بتشغيل full_pipeline الخاص بك لتحويل البيانات
# (فقط قم باستدعاء transform وليس fit_transform - أنت لا تريد أن تلائم مجموعة الاختبار!) ، وقم بتقييم النموذج النهائي في مجموعة الاختبار

final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

final_rmse

#قد ترغب في الحصول على فكرة عن مدى دقة هذا التقدير.
# لهذا ، يمكنك حساب فاصل ثقة 95٪   confidence interval لخطأ rmse باستخدام scipy.stats.t.interval :

from scipy import stats

confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                         loc=squared_errors.mean(),
                         scale=stats.sem(squared_errors)))