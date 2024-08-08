from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, lower, regexp_extract, split

from pyspark.ml import Transformer
from pyspark.ml.util import Identifiable
from pyspark.sql.functions import lit
import brtdl

class AddVersionColumn(Transformer, Identifiable):
    def __init__(self):
        super(AddVersionColumn, self).__init__()
        self.version_value = brtdl.__version__
        
    def _transform(self, df):
        return df.withColumn("version", lit(self.version_value))

spark = SparkSession.builder.appName(
    "Ch02 - Analyzing the vocabulary of Pride and Prejudice."
).getOrCreate()

book = spark.read.text("/opt/spark/data/pride-and-prejudice.txt")

lines = book.select(split(col("value"), " ").alias("line"))

words = lines.select(explode(col("line")).alias("word"))

words_lower = words.select(lower(col("word")).alias("word_lower"))
words_clean = words_lower.select(
    regexp_extract(col("word_lower"), "[a-z]*", 0).alias("word")
)
words_nonull = words_clean.where(col("word") != "")

results = words_nonull.groupby(col("word")).count()

# Create and apply the custom transformer
version_transformer = AddVersionColumn()
results = version_transformer.transform(results)

results.orderBy(col("count").desc()).show(10)