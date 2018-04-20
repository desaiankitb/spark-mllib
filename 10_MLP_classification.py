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
>>> iviris_df
# DataFrame[sepal_length: double, sepal_width: double, petal_length: double, petal_width: double, species: string, features: vector, label: double]
>>> iviris_df.take(1)
# [Row(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, species='Iris-setosa', features=DenseVector([5.1, 3.5, 1.4, 0.2]), label=0.0)]
>>> iviris_df.show(10,False)
# +------------+-----------+------------+-----------+-----------+-----------------+-----+
# |sepal_length|sepal_width|petal_length|petal_width|species    |features         |label|
# +------------+-----------+------------+-----------+-----------+-----------------+-----+
# |5.1         |3.5        |1.4         |0.2        |Iris-setosa|[5.1,3.5,1.4,0.2]|0.0  |
# |4.9         |3.0        |1.4         |0.2        |Iris-setosa|[4.9,3.0,1.4,0.2]|0.0  |
# |4.7         |3.2        |1.3         |0.2        |Iris-setosa|[4.7,3.2,1.3,0.2]|0.0  |
# |4.6         |3.1        |1.5         |0.2        |Iris-setosa|[4.6,3.1,1.5,0.2]|0.0  |
# |5.0         |3.6        |1.4         |0.2        |Iris-setosa|[5.0,3.6,1.4,0.2]|0.0  |
# |5.4         |3.9        |1.7         |0.4        |Iris-setosa|[5.4,3.9,1.7,0.4]|0.0  |
# |4.6         |3.4        |1.4         |0.3        |Iris-setosa|[4.6,3.4,1.4,0.3]|0.0  |
# |5.0         |3.4        |1.5         |0.2        |Iris-setosa|[5.0,3.4,1.5,0.2]|0.0  |
# |4.4         |2.9        |1.4         |0.2        |Iris-setosa|[4.4,2.9,1.4,0.2]|0.0  |
# |4.9         |3.1        |1.5         |0.1        |Iris-setosa|[4.9,3.1,1.5,0.1]|0.0  |
# +------------+-----------+------------+-----------+-----------+-----------------+-----+
# only showing top 10 rows

>>> splits = iviris_df.randomSplit([0.6,0.4],1)
>>> train_df = splits[0]
>>> test_df = splits[1]
>>> train_df.count()
# 92
>>> test_df.count()
# 58
>>> iviris_df.count()
# 150

>>> from pyspark.ml.classification import MultilayerPerceptronClassifier
>>> from pyspark.ml.evaluation import MulticlassClassificationEvaluator

>>> layers = [4,5,5,3]
>>> mlp = MultilayerPerceptronClassifier(layers=layers, seed=1)
>>> mlp_model = mlp.fit(train_df)
>>> mlp_predictions = mlp_model.transform(test_df)
>>> mlp_evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
>>> mlp_accuracy = mlp_evaluator.evaluate(mlp_predictions)
>>> mlp_accuracy
0.9482758620689655

