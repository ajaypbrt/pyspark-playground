from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, lower, regexp_extract, split

import os
os.environ['PYSPARK_PYTHON'] = "./environment/bin/python"
spark = SparkSession.builder.appName(
    "Ch02 - Analyzing the vocabulary of Pride and Prejudice."
).config(
    "spark.archives",
    "/opt/spark/data/brtdl.tar.gz#environment"
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

results.orderBy(col("count").desc()).show(10)
