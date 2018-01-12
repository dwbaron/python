2018-01-08

官方示例和源码
`csv_example.py`

`preProcess`数据预处理函数，正则做初步文本清理

`field`定义dedupe需要计算的字段好字段类型
```python
    fields = [
        {'field': 'Site name', 'type': 'String'},
        {'field': 'Address', 'type': 'String'},
        {'field': 'Zip', 'type': 'Exact', 'has missing': True},
        {'field': 'Phone', 'type': 'String', 'has missing': True},
    ]
```
`deduper.sample()`从数据中抽取训练集

>重点是`deduper.train()`

见下图，`self.data_model.distance()`
```python

def train(self, recall=0.95, index_predicates=True):
	# 获得训练数据的特征和目标变量
	examples, y = flatten_training(self.training_pairs)
	self.classifier.fit(self.data_model.distances(examples), y)           
	self._trainBlocker(recall, index_predicates)

```
1. `distance()`函数会根据匹配对的不同字段类型分别采用不同的距离计算方法，由`compare()`具体执行，
	封装在`DataModel`的`_field_comparators`方法里
```python
    @property 
    def _field_comparators(self) :
        start = 0
        stop = 0
        comparators = []
        for field in self.primary_fields :
            stop = start + len(field) 
            comparators.append((field.field, field.comparator, start, stop))
            start = stop

        return comparators
```
2. `self.primary_fields`定义如下
```python
class DataModel(object) :

    def __init__(self, fields):

        primary_fields, variables = typifyFields(fields)
        self.primary_fields = primary_fields
        self._derived_start = len(variables)
```
3. `primary_fields`是`typifyFields`返回的第一个值
在`typifyFields`中
```python
field_object = field_class(definition)
primary_fields.append(field_object)
```
4. `field_class`由`FIELD_CLASSES`定义
```python
FIELD_CLASSES = {k : v for k, v in base.allSubclasses(base.FieldType) if k}
```
5. 一路追溯到`FieldType`及其子类，事实上，更具体的方法在例如`variables.string`这些地方，我们终于要看到了针对字符型变量的具体方法，他们都是`FieldType`的子类
```python
crfEd = CRFEditDistance()

base_predicates = (predicates.wholeFieldPredicate,
                   predicates.firstTokenPredicate,
                   predicates.commonIntegerPredicate,
                   predicates.nearIntegersPredicate,
                   predicates.firstIntegerPredicate,
                   predicates.sameThreeCharStartPredicate,
                   predicates.sameFiveCharStartPredicate,
                   predicates.sameSevenCharStartPredicate,
                   predicates.commonTwoTokens,
                   predicates.commonThreeTokens,
                   predicates.fingerprint,
                   predicates.oneGramFingerprint,
                   predicates.twoGramFingerprint,
                   predicates.sortedAcronym)


class BaseStringType(FieldType) :
    type = None
    _Predicate = predicates.StringPredicate


    def __init__(self, definition) :
        super(BaseStringType, self).__init__(definition)

        self.predicates += indexPredicates((predicates.LevenshteinCanopyPredicate,
                                            predicates.LevenshteinSearchPredicate),
                                           (1, 2, 3, 4),
                                           self.field)

    


class ShortStringType(BaseStringType) :
    type = "ShortString"

    _predicate_functions = (base_predicates 
                            + (predicates.commonFourGram,
                               predicates.commonSixGram,
                               predicates.tokenFieldPredicate,
                               predicates.suffixArray,
                               predicates.doubleMetaphone,
                               predicates.metaphoneToken))

    _index_predicates = (predicates.TfidfNGramCanopyPredicate, 
                         predicates.TfidfNGramSearchPredicate)
    _index_thresholds = (0.2, 0.4, 0.6, 0.8)


    def __init__(self, definition) :
        super(ShortStringType, self).__init__(definition)

        if definition.get('crf', False) == True :
            self.comparator = crfEd
        else :
            self.comparator = affineGap
```
6. 进一步挖掘，看到了`CRFEditDistance`，具体方法是`Hacrf`
注意`StringPairFeatureExtractor`
```python
class CRFEditDistance(object) :
    def __init__(self) :
        classes = ['match', 'non-match']
        self.model = Hacrf(l2_regularization=100.0,
                           state_machine=DefaultStateMachine(classes))
        self.model.parameters = np.array(
            [[-0.22937526,  0.51326066],
             [ 0.01038001, -0.13348901],
             [-0.03062821,  0.13769178],
             [ 0.02024813, -0.01835538],
             [ 0.09208272,  0.15466022],
             [-0.08170265, -0.02484392],
             [-0.01762858,  0.17504624],
             [ 0.02800866, -0.04442708]],
            order='F')
        self.parameters = self.model.parameters.T
        self.model.classes = ['match', 'non-match']

        self.feature_extractor = StringPairFeatureExtractor(match=True,
                                                            numeric=False)
```
7. `Hacrf`是`pyhacrf`包里面的方法，包含了`StringPairFeatureExtractor`
我们看看他的定义，通过矩阵，对应位置字符相同的地方赋值为1
```python
class StringPairFeatureExtractor(PairFeatureExtractor):
    """ Extract features from sequence pairs.

    A grid is constructed for each sequence pair, for example for ("kaas", "cheese"):

     s * . . . @ .
     a * . . . . .
     a * . . . . .
     k * * * * * *
       c h e e s e

    For each element in the grid, a feature vector is constructed. The elements in the feature
    vector are determined by which features are active at that position in the grid. So for the
    example above, the 'match' feature will be 0 in every vector in every position except the
    position indicated with '@', where it will be 1. The 'start' feature will be 1 in all the
    positions with '*' and 0 everywhere else.


    Parameters
    ----------
    bias: float: optional (default=1.0)
        A bias term that is always added to every position in the lattice.

    start: boolean: optional
        Binary feature that activates at the start of either sequence.

    end: boolean: optional
        Binary feature that activates at the end of either sequence.

    match: boolean: optional
        Binary feature that activates when elements at a position are equal.

    numeric: boolean, optional
        Binary feature that activates when all elements at a position are numerical.

    transition: boolean, optional
        Adds binary features for pairs of (lower case) input characters.
    """
    
```
8. 此外，在两个对比字符串`s1`和`s2`组成的特征矩阵中，`s1`和`s2`的首尾处均赋值1，注意`...`的用法
```python
def starts(s1, s2) :
    M = np.zeros((s1.size, s2.size))
    M[0,...] = 1
    M[...,0] = 1
    return M

def ends(s1, s2) :
    M = np.zeros((s1.size, s2.size))
    M[(s1.size-1),...] = 1
    M[...,(s2.size-1)] = 1
    return M

def matches(s1, s2) :
    return (s1 == s2)
```
9. 在计算连续型特征时，给定特征矩阵$M=[size1, size2, n\_features]$，那么计算两个字符串序列相似度值赋值给$M[..., k]$(第k个特征)， 计算离散特征时，只找两个字符串序列对应相同字符在矩阵的位置，于是接着连续型特征之后，继续赋值，将矩阵对应相同字符的位置赋值为1；
$e.g.$ 特征包含连续型和离散型，分散在特征矩阵$M[i,j,z]$的第三个维度上，假设已经计算完$n$个连续型特征，那么计算第$m$个离散特征时，先用特征计算函数计算出对应离散特征的位置值，比如计算得到位置值$p = feature\_function(i, j, m)$，则有$M[i, j, n + p]=1$
```python
    def _extract_features(self, array1, array2):
        """ Helper to extract features for one data point. """

        feature_array = np.zeros((array1.size, array2.size, self.K),
                                 dtype='float64')

        for k, feature_function in enumerate(self._binary_features):
            feature_array[..., k] = feature_function(array1, array2)

        if self._sparse_features:
            array1 = array1.T[0]
            array2 = array2[0]
            n_binary_features = len(self._binary_features)

            for i, j in np.ndindex(array1.size, array2.size):
                k = n_binary_features

                for feature_function, num_features in self._sparse_features:
                    
                    feature_array[i, j, k + feature_function(i, j, array1, array2)] = 1.0
                    k += num_features

        return feature_array
```
10. 计算匹配以及匹配概率时具体用到了`pyhacrf`模块，类似于`sklearn`的API，是一个可训练的编辑距离计算框架：
https://github.com/dirko/pyhacrf
计算框架包含了以上特征变换的工作，用到了`state-machine`来描述在`Levenshtein`编辑距离算法之下，两个字符串序列的对齐$a$，用一个四元组$(e, ix, iy, q)$来表示，具体来讲
$e$：编辑操作（插入字符，删除字符等），$ix, iy$：两个序列发生操作的位置，$q$：编辑状态，由操作顺序和$e$来描述，通过计算CRF:
$p(a|x, y)=\frac{1}{Z_{x,y}}\prod_{i=1}^{|a|}\Phi(a_{i-1}, a_i, x, y)$
我们希望最大化对齐条件概率，$\Phi(\bullet)$是非负函数，也叫做`potential`，$a_{i-1}, a_i$类似于通过不同的操作，两个相邻对齐之间的状态转移，参数的学习通过`EM`算法（期望-最大化）来实现更迭。

---
2018-01-14

`mysql`
1. 涉及数据导出时，遇到`ERROR 1290 (HY000):–secure-file-priv`权限问题
查看安全权限

```sql
show variables like '%secure%';
```


- 发现`–secure-file-priv`对应值为`NULL`，
- 于是`net stop mysql`关闭mysql服务，
- 修改`my.ini`配置文件，添加`–secure-file-priv=''`，允许权限
- 重启`net start mysql`;

导出为csv格式代码
```sql
select * from customers into outfile 'D:/tmp/customers.csv' 
fields terminated by ',' 
optionally enclosed by '"' 
escaped by '"' 
lines terminated by '\r\n';
```