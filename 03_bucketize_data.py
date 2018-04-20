>>> from pyspark.ml.feature import Bucketizer
>>> splits = [-float("inf"), -10.0, 0.0, 10.0, float("inf")]
>>> b_data=[(-800.0,),(-10.5,),(-1.7,),(0.0,),(8.2,),(90.1,)]
>>> b_df = spark.createDataFrame(b_data, ["features"])
>>> b_df.show()
+--------+
|features|
+--------+
|  -800.0|
|   -10.5|
|    -1.7|
|     0.0|
|     8.2|
|    90.1|
+--------+

>>> bucketizer = Bucketizer(splits=splits, inputCol="features", outputCol="bfeatures")
>>> bucketed_df = bucketizer.transform(b_df)
>>> bucketed_df
DataFrame[features: double, bfeatures: double]
>>> bucketed_df.show()
+--------+---------+
|features|bfeatures|
+--------+---------+
|  -800.0|      0.0|
|   -10.5|      0.0|
|    -1.7|      1.0|
|     0.0|      2.0|
|     8.2|      2.0|
|    90.1|      3.0|
+--------+---------+
