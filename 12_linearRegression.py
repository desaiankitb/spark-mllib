>>> from pyspark.ml.regression import LinearRegression
>>> pp_df = spark.read.csv("file:///home/desaiankitb/Code/spark-ml/power-plant.csv",header=True,inferSchema=True)
>>> pp_df
# DataFrame[AT: double, V: double, AP: double, RH: double, PE: double]
>>> from pyspark.ml.feature import VectorAssembler
>>> vectorAssembler = VectorAssembler(inputCols=["AT","V","AP","RH"],outputCol="features")
>>> vpp_df = vectorAssembler.transform(pp_df)
>>> vpp_df.take(1)
# [Row(AT=14.96, V=41.76, AP=1024.07, RH=73.17, PE=463.26, features=DenseVector([14.96, 41.76, 1024.07, 73.17]))]
>>> lr = LinearRegression(featuresCol="features",labelCol="PE")
>>> lr_model = lr.fit(vpp_df)
# 2018-04-21 11:01:37 WARN  WeightedLeastSquares:66 - regParam is zero, which might cause numerical instability and overfitting.
# 2018-04-21 11:01:37 WARN  BLAS:61 - Failed to load implementation from: com.github.fommil.netlib.NativeSystemBLAS
# 2018-04-21 11:01:37 WARN  BLAS:61 - Failed to load implementation from: com.github.fommil.netlib.NativeRefBLAS
# 2018-04-21 11:01:37 WARN  LAPACK:61 - Failed to load implementation from: com.github.fommil.netlib.NativeSystemLAPACK
# 2018-04-21 11:01:37 WARN  LAPACK:61 - Failed to load implementation from: com.github.fommil.netlib.NativeRefLAPACK
>>> lr_model.coefficients
# DenseVector([-1.9775, -0.2339, 0.0621, -0.1581])
>>> lr_model.intercept
# 454.6092744523414
>>> lr_model.summary.rootMeanSquaredError
# 4.557126016749488
>>> lr_model.save("file:///home/desaiankitb/Code/spark-ml/lr1.model")
