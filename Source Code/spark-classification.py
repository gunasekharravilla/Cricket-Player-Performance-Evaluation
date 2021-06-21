import pyspark
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("Sachin Tendulkar Performance Analysis") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()
input_data = spark.read.csv('file:///home/edureka/Downloads/customer_churn.csv', header=True, inferSchema=True)

test_data = spark.read.csv('file:///home/edureka/Downloads/new_customers.csv', header=True, inferSchema=True)
input_data.printSchema()
test_data.printSchema()
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=['Age', 'Total_Purchase', 'Account_Manager', 'Years', 'Num_Sites'],
                            outputCol='features')
output_data = assembler.transform(input_data)
output_data.printSchema()
final_data = output_data.select('features', 'churn')
train, test = final_data.randomSplit([0.7, 0.3])
model = LogisticRegression(labelCol='churn')
model = model.fit(train)
summary = model.summary
summary.predictions.describe().show()
from pyspark.ml.evaluation import BinaryClassificationEvaluator
predictions = model.evaluate(test)
evaluator = BinaryClassificationEvaluator(rawPredictionCol='prediction', labelCol='churn')
evaluator.evaluate(predictions.predictions)
model1 = LogisticRegression(labelCol='churn')
model1 = model1.fit(final_data)
test_data = assembler.transform(test_data)
results = model1.transform(test_data)
results.select('Company', 'prediction').show()