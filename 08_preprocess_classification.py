>>> from pyspark.sql.functions import *
>>> from pyspark.ml.feature import VectorAssembler
>>> from pyspark.ml.feature import StringIndexer


>>> iris_df = spark.read.csv("file:///home/ankit/Documents/personal/code/spark-ml/iris.data", inferSchema=True)
>>> iris_df
# DataFrame[_c0: double, _c1: double, _c2: double, _c3: double, _c4: string]
>>> iris_df.take(1)
[Row(_c0=5.1, _c1=3.5, _c2=1.4, _c3=0.2, _c4=u'Iris-setosa')]

>>> iris_df = iris_df.select(col("_c0").alias("sepal_length"), \
   col("_c1").alias("sepal_width"), \
   col("_c2").alias("petal_length"), \
   col("_c3").alias("petal_width"), \
   col("_c4").alias("species"))
>>> iris_df.take(1)
# [Row(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, species=u'Iris-setosa')]
>>> vectorAssembler = VectorAssembler(inputCols=["sepal_length","sepal_width","petal_length","petal_width"], outputCol="features")
>>> viris_df = vectorAssembler.transform(iris_df)
>>> viris_df.take(1)
# [Row(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, species=u'Iris-setosa', features=DenseVector([5.1, 3.5, 1.4, 0.2]))]
>>> indexer = StringIndexer(inputCol="species", outputCol="label")
>>> iviris_df = indexer.fit(viris_df).transform(viris_df)
>>> iviris_df.show(1)
# +------------+-----------+------------+-----------+-----------+-----------------+-----+
# |sepal_length|sepal_width|petal_length|petal_width|    species|         features|label|
# +------------+-----------+------------+-----------+-----------+-----------------+-----+
# |         5.1|        3.5|         1.4|        0.2|Iris-setosa|[5.1,3.5,1.4,0.2]|  0.0|
# +------------+-----------+------------+-----------+-----------+-----------------+-----+
# only showing top 1 row
