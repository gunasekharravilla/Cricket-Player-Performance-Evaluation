from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint

# Load and parse the data
from pyspark.sql import SparkSession
ss = SparkSession \
    .builder \
    .appName("Sachin Tendulkar Performance Analysis") \
    .config("spark.some.config.option", "some-value")
from pyspark import SparkContext
#logFile = "file:///home/hadoop/spark-2.1.0-bin-hadoop2.7/README.md"
sc = SparkContext("local", "first app")

def parsePoint(line):
    values = [float(x) for x in line.split(' ')]
    print(values)
    return LabeledPoint(values[0], values[1:])
data = sc.textFile("cleaned-modified-new.txt")
parsedData = data.map(parsePoint)

# Build the model
model = SVMWithSGD.train(parsedData, iterations=100)

# Evaluating the model on training data
labelsAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
trainErr = labelsAndPreds.filter(lambda lp: lp[0] != lp[1]).count() / float(parsedData.count())
print("Training Error = " + str(trainErr))

# Save and load model
#model.save(sc, "pythonSVMWithSGDModel")
#sameModel = SVMModel.load(sc, "pythonSVMWithSGDModel")
