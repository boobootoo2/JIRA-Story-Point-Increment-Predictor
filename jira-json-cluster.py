# Databricks notebook source
# Attached to: jira-json-cluster

# Your actual Python code goes here
# Make sure to write valid Python syntax in this section

# COMMAND ----------

from pyspark.sql.functions import col

data_dir = "/Volumes/jira_story_point_estimator/default/jira_data_partitions"
partition_dirs = [f.path for f in dbutils.fs.ls(data_dir) if f.isDir()]

df = (
    spark.read
    .option("multiline", "true")
    .json([f"{d}/*.json" for d in partition_dirs])
    .repartition(5)
)

display(df)

# COMMAND ----------

df_transformed = (
    df.select(
        col("key"),
        col("fields.project.key").alias("project"),
        col("fields.created"),
        col("fields.resolutiondate"),
        col("fields.summary"),
        col("fields.description")
    )
    .dropna()
)

display(df_transformed)

# COMMAND ----------

df_transformed = (
    df.select(
        col("key"),
        col("fields.project.key").alias("project"),
        col("fields.created"),
        col("fields.resolutiondate"),
        col("fields.summary"),
        col("fields.description")
    )
    .dropna()
)

display(df_transformed)

# COMMAND ----------

row_count = df_transformed.count()
col_count = len(df_transformed.columns)
print(f"Rows: {row_count}, Columns: {col_count}")

# COMMAND ----------

from pyspark.sql.functions import col, regexp_replace

df_transformed_no_tz = df_transformed.withColumn(
    "created", regexp_replace(col("created"), r"\+\d$", "")
).withColumn(
    "resolutiondate", regexp_replace(col("resolutiondate"), r"\+\d$", "")
)

display(df_transformed_no_tz)

# COMMAND ----------

from pyspark.sql.functions import to_timestamp, col, unix_timestamp, regexp_replace

# Remove timezone information from the date fields
df_transformed_cleaned = df_transformed.withColumn(
    "created_cleaned", regexp_replace(col("created"), r"\+\d+$", "")
).withColumn(
    "resolutiondate_cleaned", regexp_replace(col("resolutiondate"), r"\+\d+$", "")
)

# Convert cleaned date fields to Unix timestamps
df_transformed_unix_timestamp = df_transformed_cleaned.withColumn(
    "created_unix", unix_timestamp(col("created_cleaned"), "yyyy-MM-dd'T'HH:mm:ss.SSS")
).withColumn(
    "resolutiondate_unix", unix_timestamp(col("resolutiondate_cleaned"), "yyyy-MM-dd'T'HH:mm:ss.SSS")
)

display(df_transformed_unix_timestamp)

# COMMAND ----------

from pyspark.sql.functions import col, expr

df_with_cycle_hours = (
    df_transformed_unix_timestamp
    .repartition(5)
    .withColumn(
        "cycle_hours",
        expr("ROUND((resolutiondate_unix - created_unix) / 3600, 0)")
    )
)

display(df_with_cycle_hours)

# COMMAND ----------

from pyspark.sql.functions import col, lit, array
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, DoubleType

# Assemble features for KMeans
assembler = VectorAssembler(
    inputCols=["cycle_hours"],
    outputCol="features"
)
df_features = assembler.transform(df_with_cycle_hours)

# Get distinct project list
projects = [
    row['project']
    for row in df_features.select("project").distinct().collect()
]

results = []
for project in projects:
    df_proj = df_features.filter(col("project") == project)
    if df_proj.count() >= 8:  # KMeans requires at least k rows
        kmeans = KMeans(
            k=8,
            seed=42,
            featuresCol="features"
        )
        model = kmeans.fit(df_proj)
        centroids = [float(c[0]) for c in model.clusterCenters()]
        df_proj = df_proj.withColumn("cycle_centroids", lit(centroids))
        results.append(df_proj)
    else:
        # Not enough data for clustering, fill with empty list
        df_proj = df_proj.withColumn("cycle_centroids", array())
        results.append(df_proj)

# Union all project DataFrames
from functools import reduce

df_with_centroids = reduce(
    lambda df1, df2: df1.unionByName(df2),
    results
)

# Use Python's built-in round, not pyspark.sql.functions.round
def round_centroids(centroids):
    return [float(round(float(c))) for c in centroids] if centroids else []

round_centroids_udf = udf(
    round_centroids,
    ArrayType(DoubleType())
)

df_with_centroids_rounded = df_with_centroids.withColumn(
    "cycle_centroids_rounded",
    round_centroids_udf(col("cycle_centroids"))
)

display(
    df_with_centroids_rounded.select(
        "project",
        "cycle_hours",
        "cycle_centroids_rounded"
    )
)

# COMMAND ----------

from pyspark.sql.functions import array_sort, map_from_arrays, array, lit

# Define the Fibonacci series as a Spark array
fibonacci = array([lit(x) for x in [1, 2, 3, 5, 8, 13, 21, 34]])
fib_length = 8

# Sort centroids for each row and pad/truncate to match Fibonacci length
df_with_sorted_centroids = df_with_centroids_rounded.withColumn(
    "sorted_centroids",
    array_sort(col("cycle_centroids_rounded"))
).withColumn(
    "centroids_padded",
    expr(
        f"""
        CASE 
            WHEN size(sorted_centroids) >= {fib_length} THEN slice(sorted_centroids, 1, {fib_length})
            ELSE concat(
                sorted_centroids, 
                array_repeat(sorted_centroids[array_max(transform(sequence(1, size(sorted_centroids)), x -> x))], {fib_length} - size(sorted_centroids))
            )
        END
        """
    )
).withColumn(
    "fib_to_centroid",
    map_from_arrays(fibonacci, col("centroids_padded"))
)

display(
    df_with_sorted_centroids.select(
        "project", "cycle_hours", "cycle_centroids_rounded", "fib_to_centroid"
    )
)

# COMMAND ----------

from pyspark import StorageLevel

df_with_sorted_centroids.persist(StorageLevel.MEMORY_AND_DISK)
df_with_sorted_centroids

# COMMAND ----------

display(df_with_sorted_centroids)

# COMMAND ----------

df_with_sorted_centroids.write.mode("overwrite").saveAsTable("df_with_sorted_centroids")

# COMMAND ----------

df_with_sorted_centroids.write.mode("overwrite").parquet("/mnt/backup/df_with_sorted_centroids_parquet")

# COMMAND ----------

df_with_sorted_centroids = spark.table("df_with_sorted_centroids")

# COMMAND ----------

from pyspark.sql.functions import udf, col
from pyspark.sql.types import IntegerType

def find_increment(cycle_hours, fib_to_centroid):
    if not fib_to_centroid or cycle_hours is None:
        return None
    # Get sorted keys as strings (since Spark map keys are strings)
    fib_keys = sorted(fib_to_centroid.keys(), key=lambda x: int(x))
    # Filter out keys where the centroid value is None
    valid_keys = [k for k in fib_keys if fib_to_centroid[k] is not None]
    if not valid_keys:
        return None
    # Find the closest centroid and its key
    closest_key = min(
        valid_keys,
        key=lambda k: abs(cycle_hours - fib_to_centroid[k])
    )
    # Return the 1-based index (increment) of the closest key
    return fib_keys.index(closest_key) + 1

find_increment_udf = udf(find_increment, IntegerType())

df_with_actual_increment = df_with_sorted_centroids.withColumn(
    "actual_increment",
    find_increment_udf(col("cycle_hours"), col("fib_to_centroid"))
)

display(df_with_actual_increment.select(
    "project", "cycle_hours", "fib_to_centroid", "actual_increment"
))

# COMMAND ----------

df_with_actual_increment.write.mode("overwrite").saveAsTable("df_with_actual_increment")

# COMMAND ----------

from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col

tokenizer = Tokenizer(
    inputCol="description",
    outputCol="desc_tokens"
)
df_tokens = tokenizer.transform(df_with_actual_increment)

hashing_tf = HashingTF(
    inputCol="desc_tokens",
    outputCol="desc_tf",
    numFeatures=500
)
df_tf = hashing_tf.transform(df_tokens)

idf = IDF(
    inputCol="desc_tf",
    outputCol="desc_tfidf"
)
idf_model = idf.fit(df_tf)
df_tfidf = idf_model.transform(df_tf)

df_tfidf = df_tfidf.withColumn(
    "desc_tfidf_arr",
    vector_to_array(col("desc_tfidf"))
)

for i in range(500):
    df_tfidf = df_tfidf.withColumn(
        f"desc_dim_{i+1}",
        col("desc_tfidf_arr")[i]
    )

df_tfidf = df_tfidf.drop(
    "desc_tokens",
    "desc_tf",
    "desc_tfidf",
    "desc_tfidf_arr"
)

display(df_tfidf)

# COMMAND ----------

num_rows = df_tfidf.count()
num_cols = len(df_tfidf.columns)
print(f"Rows: {num_rows}, Columns: {num_cols}")

# COMMAND ----------

df_tfidf.write.mode("overwrite").option("mergeSchema", "true").saveAsTable("df_tfidf")

# COMMAND ----------

from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col

# Tokenize the summary column
tokenizer = Tokenizer(inputCol="summary", outputCol="summary_tokens")
df_tokens = tokenizer.transform(df_tfidf)

# Apply HashingTF to convert tokens into numerical vectors
hashing_tf = HashingTF(inputCol="summary_tokens", outputCol="summary_tf", numFeatures=500)
df_tf = hashing_tf.transform(df_tokens)

# Apply IDF to the HashingTF output
idf = IDF(inputCol="summary_tf", outputCol="summary_tfidf")
idf_model = idf.fit(df_tf)
df_tfidf_with_summary = idf_model.transform(df_tf)

# Convert the TF-IDF vector to an array
df_tfidf_with_summary = df_tfidf_with_summary.withColumn("summary_tfidf_arr", vector_to_array(col("summary_tfidf")))

# Add 200 summary_dim_{i+1} columns
for i in range(500):
    df_tfidf_with_summary = df_tfidf_with_summary.withColumn(f"summary_dim_{i+1}", col("summary_tfidf_arr")[i])

# Drop intermediate columns
df_tfidf_with_summary = df_tfidf_with_summary.drop("summary_tokens", "summary_tf", "summary_tfidf", "summary_tfidf_arr")

display(df_tfidf_with_summary)

# COMMAND ----------

df_tfidf_with_summary.write.mode("overwrite").option("mergeSchema", "true").saveAsTable("df_tfidf_with_summary")

# COMMAND ----------

df_tfidf_with_summary.write.mode("overwrite").parquet("/mnt/backup/df_tfidf_with_summary_parquet")

# COMMAND ----------

df_tfidf_with_summary = spark.table("df_tfidf_with_summary")

# COMMAND ----------

from pyspark.sql.functions import col, lit

# List all dimension columns
desc_dim_cols = [f"desc_dim_{i+1}" for i in range(500)]
summary_dim_cols = [f"summary_dim_{i+1}" for i in range(500)]
all_dim_cols = desc_dim_cols + summary_dim_cols

# Ensure all dimension columns exist, fill missing with 0
for dim in all_dim_cols:
    if dim not in df_tfidf_with_summary.columns:
        df_tfidf_with_summary = df_tfidf_with_summary.withColumn(dim, lit(0))

# Compute Spearman correlations
correlations = []
for dim in all_dim_cols:
    corr = df_tfidf_with_summary.stat.corr("actual_increment", dim)
    correlations.append((dim, abs(corr)))

# Get top 100 dimensions by absolute correlation
top_dims = sorted(correlations, key=lambda x: x[1], reverse=True)[:100]
top_dim_names = [dim for dim, _ in top_dims]

# Select required columns
df_top = df_tfidf_with_summary.select(
    ["key", "project", "actual_increment"] + top_dim_names
)

display(df_top)

# COMMAND ----------

import pandas as pdimport seaborn as sns

import matplotlib.pyplot as plt

# List all dimension columns with prefixes 'summary_dim_' or 'desc_dim_'
all_dim_cols = [
    col for col in df_tfidf_with_summary.columns
    if col.startswith('summary_dim_') or col.startswith('desc_dim_')
]

# Ensure all dimension columns exist, fill missing with 0
from pyspark.sql.functions import lit
for dim in all_dim_cols:
    if dim not in df_tfidf_with_summary.columns:
        df_tfidf_with_summary = df_tfidf_with_summary.withColumn(dim, lit(0))

# Compute Spearman correlations
correlations = []
for dim in all_dim_cols:
    corr = df_tfidf_with_summary.stat.corr("actual_increment", dim)
    correlations.append((dim, abs(corr)))

# Get top 20 dimensions by absolute correlation
top_dims = sorted(correlations, key=lambda x: x[1], reverse=True)[:20]
top_dim_names = [dim for dim, _ in top_dims]

# Prepare data for heatmap: a single row with correlations
heatmap_df = (
    pd.DataFrame([dict(top_dims)], index=['Correlation'])
    .reindex(columns=top_dim_names)
)

plt.figure(figsize=(20, 2))
sns.heatmap(
    heatmap_df,
    cmap='coolwarm',
    annot=True,
    fmt=".2f",
    linewidths=.5,
    cbar=True
)
plt.title('Correlation Heatmap: Top 50 Dimensions vs actual_increment')
plt.yticks(rotation=0)
plt.show()

# COMMAND ----------

import pandas as pd

# Sort and get top 20 correlations
top_20 = sorted(correlations, key=lambda x: x[1], reverse=True)[:20]
top_20_df = pd.DataFrame(top_20, columns=["Dimension", "AbsCorrelation"])

display(top_20_df)

# COMMAND ----------

train_df, test_df = df_tfidf_with_summary.randomSplit([0.8, 0.2], seed=42)
display(train_df)
display(test_df)

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

# Select X and Y columns
X_cols = all_dim_cols
Y_col = "actual_increment"

# Drop the existing "features" column if it exists
df_tfidf_with_summary = df_tfidf_with_summary.drop("features")

# Create a VectorAssembler for X
assembler = VectorAssembler(inputCols=X_cols, outputCol="features")
df_assembled = assembler.transform(df_tfidf_with_summary)

# Split the data into training and testing sets
train_df, test_df = df_assembled.randomSplit([0.8, 0.2], seed=42)

# COMMAND ----------

# Filter out rows with nulls in label or prediction columns before evaluation
predictions_clean = predictions.filter(
    (predictions[Y_col].isNotNull()) & (predictions["prediction"].isNotNull())
)

# Evaluate the model
evaluator = RegressionEvaluator(
    labelCol=Y_col,
    predictionCol="prediction",
    metricName="r2"
)
r2 = evaluator.evaluate(predictions_clean)

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

# Evaluate the model using RMSE
evaluator = RegressionEvaluator(labelCol=Y_col, predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions.na.drop())

# COMMAND ----------

# MAGIC %pip install azure-ai-textanalytics azure-core
# MAGIC
# MAGIC from pyspark.ml.evaluation import RegressionEvaluator
# MAGIC from pyspark.sql.functions import abs as sql_abs, col, when
# MAGIC
# MAGIC # Filter out rows with nulls in label or prediction columns
# MAGIC predictions_clean = predictions.filter(
# MAGIC     col(Y_col).isNotNull() & col("prediction").isNotNull()
# MAGIC )
# MAGIC
# MAGIC # R^2 calculation
# MAGIC evaluator_r2 = RegressionEvaluator(
# MAGIC     labelCol=Y_col,
# MAGIC     predictionCol="prediction",
# MAGIC     metricName="r2"
# MAGIC )
# MAGIC r2 = evaluator_r2.evaluate(predictions_clean)
# MAGIC
# MAGIC # Percent accuracy calculation (for regression, use MAPE as a proxy)
# MAGIC predictions_clean = predictions_clean.withColumn(
# MAGIC     "ape",
# MAGIC     when(
# MAGIC         col(Y_col) != 0,
# MAGIC         sql_abs(col(Y_col) - col("prediction")) / sql_abs(col(Y_col))
# MAGIC     ).otherwise(0)
# MAGIC )
# MAGIC mape = predictions_clean.agg({"ape": "avg"}).collect()[0][0]
# MAGIC percent_accuracy = 1 - mape if mape is not None else None
# MAGIC
# MAGIC # Aggregate results as a DataFrame
# MAGIC from pyspark.sql import Row
# MAGIC agg_df = spark.createDataFrame([
# MAGIC     Row(R2=r2, PercentAccuracy=percent_accuracy)
# MAGIC ])
# MAGIC
# MAGIC display(agg_df)

# COMMAND ----------

spark.sql("DROP TABLE IF EXISTS agg_df")
agg_df.write.saveAsTable("agg_df")

# COMMAND ----------

agg_df.write.mode("overwrite").parquet("/mnt/backup/agg_df_parquet")

# COMMAND ----------

agg_df = spark.table("agg_df")

# COMMAND ----------

display(agg_df)

# COMMAND ----------

df_with_actual_increment = spark.table("jira_story_point_estimator.default.df_with_actual_increment")

# COMMAND ----------

# Repartition the DataFrame into 1000 partitions
df_partitioned = df_with_actual_increment.repartition(1000)

from pyspark.sql.functions import pandas_udf, col
from pyspark.sql.types import ArrayType, FloatType
import pandas as pd

@pandas_udf(ArrayType(FloatType()))
def embed_udf(texts: pd.Series) -> pd.Series:
    from transformers import AutoTokenizer, AutoModel
    import torch

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    embeddings = []
    for text in texts:
        if text is None or text.strip() == "":
            embeddings.append([0.0] * 384)
            continue
        inputs = tokenizer([text], padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            model_output = model(**inputs)
        emb = model_output.last_hidden_state.mean(dim=1)[0].numpy()
        # Use all 384 dimensions
        embeddings.append(emb.tolist())
    return pd.Series(embeddings)

# Apply the embedding UDF on the repartitioned DataFrame
df_embed = (
    df_partitioned
    .withColumn("desc_embed", embed_udf(col("description")))
    .withColumn("summary_embed", embed_udf(col("summary")))
)

# Expand embedding arrays into separate columns (384 dims)
for i in range(384):
    df_embed = df_embed.withColumn(f"desc_dim_{i+1}", col("desc_embed")[i])
    df_embed = df_embed.withColumn(f"summary_dim_{i+1}", col("summary_embed")[i])

df_embed = df_embed.drop("desc_embed", "summary_embed")

display(df_embed)

# COMMAND ----------

spark.sql("DROP TABLE IF EXISTS df_embed")
df_embed.write.saveAsTable("df_embed")

# COMMAND ----------

import databricks.automl

summary = databricks.automl.regress(
    dataset = spark.table("df_embed"),
    target_col = "actual_increment",
    exclude_cols = ["key", "project", "created", "resolutiondate", "summary", "description"]
)