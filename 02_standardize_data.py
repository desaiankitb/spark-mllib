# Standardize [-1, 1] (normal distribution)
# Unit varience and Zero mean
# Bell shape curve 
# some algo works better (eg. SVM)

>>> from pyspark.ml.feature import StandardScaler
>>> from pyspark.ml.linalg import Vectors
>>> df_features = spark.createDataFrame([ 
    (1, Vectors.dense([10.0, 10000.0, 1.0]),),
    (2, Vectors.dense([20.0, 30000.0, 2.0]),),
    (3, Vectors.dense([30.0, 40000.0, 3.0]),)
    ],["id","features"])
>>> df_features
DataFrame[id: bigint, features: vector]
>>> df_features.show(10,False)
+---+------------------+
|id |features          |
+---+------------------+
|1  |[10.0,10000.0,1.0]|
|2  |[20.0,30000.0,2.0]|
|3  |[30.0,40000.0,3.0]|
+---+------------------+

>>> stand_smodel = feature_stand_scaler.fit(df_features)
>>> stand_sfeatures_df = stand_smodel.transform(df_features)
>>> stand_sfeatures_df.show(10,False)
+---+------------------+------------------------------+
|id |features          |sfeatures                     |
+---+------------------+------------------------------+
|1  |[10.0,10000.0,1.0]|[-1.0,-1.091089451179962,-1.0]|
|2  |[20.0,30000.0,2.0]|[0.0,0.2182178902359923,0.0]  |
|3  |[30.0,40000.0,3.0]|[1.0,0.8728715609439696,1.0]  |
+---+------------------+------------------------------+
