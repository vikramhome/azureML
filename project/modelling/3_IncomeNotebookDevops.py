# Databricks notebook source
# MAGIC %md Azure ML & Azure Databricks notebooks by Rene Bremer (original taken from Parashar Shah)
# MAGIC 
# MAGIC Copyright (c) Microsoft Corporation. All rights reserved.
# MAGIC 
# MAGIC Licensed under the MIT License.

# COMMAND ----------

# MAGIC %md In this notebook, the level of income of a person is predicted (higher of lower than 50k per year). 
# MAGIC In this notebook, the following steps are executed:
# MAGIC 
# MAGIC 1. Initialize Azure Machine Learning Service
# MAGIC 2. Add model to Azure Machine Learning Service
# MAGIC 
# MAGIC The compact version of this notebook can be found in the same folder as Extensive_predictionIncomeLevel, in which only step 1 and step 7 are executed.

# COMMAND ----------

par_model_name= dbutils.widgets.get("model_name")

# COMMAND ----------

#import azureml.core
#from azureml.core import Workspace
#from azureml.core.authentication import ServicePrincipalAuthentication
#from azureml.core.run import Run
#from azureml.core.experiment import Experiment

# Check core SDK version number
#print("SDK version:", azureml.core.VERSION)

# COMMAND ----------

import os
import urllib
import pprint
import numpy as np
import shutil
import time
import pandas as pd

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import OneHotEncoder, OneHotEncoderEstimator, StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
#from sqlalchemy import create_engine

# COMMAND ----------

# Download AdultCensusIncome.csv from Azure CDN. This file has 32,561 rows.
# basedataurl = "https://amldockerdatasets.azureedge.net"
# datafile = "AdultCensusIncome.csv"
# datafile_dbfs = os.path.join("/dbfs", datafile)

# if os.path.isfile(datafile_dbfs):
#     print("found {} at {}".format(datafile, datafile_dbfs))
# else:
#     print("downloading {} to {}".format(datafile, datafile_dbfs))
#     urllib.request.urlretrieve(os.path.join(basedataurl, datafile), datafile_dbfs)
#     time.sleep(30)

# COMMAND ----------

# Create a Spark dataframe out of the csv file.
# data_all = sqlContext.read.format('csv').options(header='true', # inferSchema='true', ignoreLeadingWhiteSpace='true', # ignoreTrailingWhiteSpace='true').load(datafile)
# print("({}, {})".format(data_all.count(), # # len(data_all.columns)))


# inserted new code to get data from db

jdbcHostname = "40.85.172.154"
jdbcPort = "1433"
jdbcDatabase = "databaseab1"
jdbcUsername = "mladmin"
jdbcPassword = "Satyam@12345"

#// Create the JDBC URL without passing in the user and password parameters.
#jdbcUrl = s"jdbc:sqlserver://${jdbcHostname}:${jdbcPort};database=${jdbcDatabase}"

#// Create a Properties() object to hold the parameters.

driver = "com.microsoft.sqlserver.jdbc.SQLServerDriver"
#url = "jdbc:sqlserver://"+jdbcHostname+":"+jdbcPort+";database="+jdbcDatabase,
url = "jdbc:sqlserver://"+jdbcHostname+":"+jdbcPort+";databaseName="+jdbcDatabase+";user="+jdbcUsername+";password="+jdbcPassword

table = "AdultCensusIncome"


data_all = spark.read.format("jdbc")\
  .option("driver", driver)\
  .option("url", url)\
  .option("dbtable", table)\
  .load()



#engine = create_engine(url)

#with engine.connect() as conn, conn.begin():
#    data_all = pd.read_sql_table('table', conn)

#import java.util.Properties
#connectionProperties = new Properties()

#connectionProperties.put("user", s"${jdbcUsername}")
#connectionProperties.put("password", s"${jdbcPassword}")
#driverClass = "com.microsoft.sqlserver.jdbc.SQLServerDriver"
#connectionProperties.setProperty("Driver", driverClass)

#val employees_table = spark.read.jdbc(jdbcUrl, "employees", connectionProperties)

#pushdown_query = "(select * from AdultCensusIncome)"
#data_all = spark.read.jdbc(url=jdbcUrl, table=pushdown_query, properties=connectionProperties)
display(data_all)

#data_all = pd.DataFrame(data_all_p)

#display(data_all)

data_all.shape

len(data_all.index)

#renaming columns, all columns that contain a - will be replaced with an "_"
columns_new = [col.replace("-", "_") for col in data_all.columns]
data_all = data_all.toDF(*columns_new)

data_all.printSchema()

# COMMAND ----------

#data_all_p = pd.DataFrame(data_all)
#data_all['split'] = np.random.randn(data_all.shape[0], 1)
#print(data_all_p)
#columns2=["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country","income"]
data_all_p = pd.DataFrame(data_all,columns='age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-los','hours-per-week','native-country','income')
#data_all['split'] = np.random.randn(data_all.shape[0], 1)
print(data_all_p)


msk = np.random.rand(len(data_all_p)) <= 0.7

train = data_all_p[msk]
test = data_all_p[~msk]


#(trainingData, testData) = data_all.randomSplit([0.7, 0.3], seed=1223)

#(testData, trainingData) = data_all.randomSplit([0.3, 0.7], seed=1223)

display(trainingData)
trainingData.printSchema()

display(testData)
testData.printSchema()

#trainingData.shape

len(training_data.index)

#testData.shape

len(test_data.index)



# COMMAND ----------

label = "income"
dtypes = dict(trainingData.dtypes)
dtypes.pop(label)

si_xvars = []
ohe_xvars = []
featureCols = []
for idx,key in enumerate(dtypes):
    if dtypes[key] == "string":
        featureCol = "-".join([key, "encoded"])
        featureCols.append(featureCol)
        
        tmpCol = "-".join([key, "tmp"])
        # string-index and one-hot encode the string column
        #https://spark.apache.org/docs/2.3.0/api/java/org/apache/spark/ml/feature/StringIndexer.html
        #handleInvalid: Param for how to handle invalid data (unseen labels or NULL values). 
        #Options are 'skip' (filter out rows with invalid data), 'error' (throw an error), 
        #or 'keep' (put invalid data in a special additional bucket, at index numLabels). Default: "error"
        si_xvars.append(StringIndexer(inputCol=key, outputCol=tmpCol, handleInvalid="skip"))
        ohe_xvars.append(OneHotEncoder(inputCol=tmpCol, outputCol=featureCol))
    else:
        featureCols.append(key)

# string-index the label column into a column named "label"
si_label = StringIndexer(inputCol=label, outputCol='label')

# assemble the encoded feature columns in to a column named "features"
assembler = VectorAssembler(inputCols=featureCols, outputCol="features")

# COMMAND ----------

model_dbfs = os.path.join("/dbfs", par_model_name)

# COMMAND ----------

# Regularization Rates
from pyspark.ml.classification import LogisticRegression

# try a bunch of alpha values in a Linear Regression (Ridge) model
reg=0
print("Regularization rate: {}".format(reg))
# create a bunch of child runs
#with root_run.child_run("reg-" + str(reg)) as run:
# create a new Logistic Regression model.
        
lr = LogisticRegression(regParam=reg)
        
# put together the pipeline
pipe = Pipeline(stages=[*si_xvars, *ohe_xvars, si_label, assembler, lr])

# train the model
model_pipeline = pipe.fit(trainingData)
        
# make prediction
predictions = model_pipeline.transform(testData)

# evaluate. note only 2 metrics are supported out of the box by Spark ML.
bce = BinaryClassificationEvaluator(rawPredictionCol='rawPrediction')
au_roc = bce.setMetricName('areaUnderROC').evaluate(predictions)
au_prc = bce.setMetricName('areaUnderPR').evaluate(predictions)
truePositive = predictions.select("label").filter("label = 1 and prediction = 1").count()
falsePositive = predictions.select("label").filter("label = 0 and prediction = 1").count()
trueNegative = predictions.select("label").filter("label = 0 and prediction = 0").count()
falseNegative = predictions.select("label").filter("label = 1 and prediction = 0").count()

# log reg, au_roc, au_prc and feature names in run history
#run.log("reg", reg)
#run.log("au_roc", au_roc)
#run.log("au_prc", au_prc)
        
print("Area under ROC: {}".format(au_roc))
print("Area Under PR: {}".format(au_prc))
       
#    run.log("truePositive", truePositive)
#    run.log("falsePositive", falsePositive)
#    run.log("trueNegative", trueNegative)
#    run.log("falseNegative", falseNegative)
                                                                                                                                                                  
print("TP: " + str(truePositive) + ", FP: " + str(falsePositive) + ", TN: " + str(trueNegative) + ", FN: " + str(falseNegative))                                                                         
        
#    run.log_list("columns", trainingData.columns)

# save model
model_pipeline.write().overwrite().save(par_model_name)
        
# upload the serialized model into run history record
mdl, ext = par_model_name.split(".")
model_zip = mdl + ".zip"
shutil.make_archive('/dbfs/'+ mdl, 'zip', model_dbfs)
##    run.upload_file("outputs/" + par_model_name, model_zip)        
    #run.upload_file("outputs/" + model_name, path_or_stream = model_dbfs) #cannot deal with folders

    # now delete the serialized model from local folder since it is already uploaded to run history 
    #shutil.rmtree(model_dbfs)
    #os.remove(model_zip)


# COMMAND ----------

# Declare run completed
#root_run.complete()
#root_run_id = root_run.id
#print ("run id:", root_run.id)

# COMMAND ----------

#Register the model already in the Model Managment of azure ml service workspace
#from azureml.core.model import Model
#mymodel = Model.register(model_path = "/dbfs/" + par_model_name, # this points to a local file
#                       model_name = par_model_name, # this is the name
#                       description = "testrbdbr",
#                       workspace = ws)
#print(mymodel.name, mymodel.id, mymodel.version, sep = '\t')

# COMMAND ----------
