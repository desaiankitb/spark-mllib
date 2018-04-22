>>> from pyspark.ml.regression import DecisionTreeRegressor
>>> from pyspark.ml.evaluation import RegressionEvaluator
>>> from pyspark.ml.feature import VectorAssembler 
>>> pp_df = spark.read.csv("file:///home/desaiankitb/Code/spark-ml/power-plant.csv",header=True,inferSchema=True)
>>> pp_df.take(1)
# [Row(AT=14.96, V=41.76, AP=1024.07, RH=73.17, PE=463.26)]
>>> vectorAssembler = VectorAssembler(inputCols=["AT","V","AP","RH"],outputCol="features")
>>> vpp_df = vectorAssembler.transform(pp_df)
>>> vpp_df.take(1)
# [Row(AT=14.96, V=41.76, AP=1024.07, RH=73.17, PE=463.26, features=DenseVector([14.96, 41.76, 1024.07, 73.17]))]
>>> splits =  vpp_df.randomSplit([0.7,0.3])
>>> train_df = splits[0]
>>> test_df =splits[1]
>>> train_df.count()
# 6747
>>> test_df.count()
# 2821
>>> vpp_df.count()
# 9568
>>> dt = DecisionTreeRegressor(featuresCol="features", labelCol="PE")
>>> dt_model = dt.fit(train_df)
>>> dt_predictions = dt_model.transform(test_df)
>>> dt_evaluator = RegressionEvaluator(labelCol="PE", predictionCol="prediction", metricName="rmse")
>>> rmse = dt_evaluator.evaluate(dt_predictions)
>>> rmse
# 4.599471665375514

>>> from pyspark.ml.regression import GBTRegressor
>>> gbt = GBTRegressor(featuresCol="features",labelCol="PE")
>>> gbt_model = gbt.fit(train_df)
>>> gbt_predictions = gbt_model.transform(test_df)
>>> gbt_evaluator = RegressionEvaluator(labelCol="PE",predictionCol="prediction", metricName="rmse")
>>> gbt_rmse = gbt_evaluator.evaluate(gbt_predictions)
>>> gbt_rmse
# 4.176500037720063