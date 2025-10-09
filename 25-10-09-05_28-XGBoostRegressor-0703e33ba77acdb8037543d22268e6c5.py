# Databricks notebook source
# MAGIC %md
# MAGIC # XGBoost Regressor training
# MAGIC - This is an auto-generated notebook.
# MAGIC - To reproduce these results, attach this notebook to a cluster with runtime version **16.4.x-cpu-ml-photon-scala2.12**, and rerun it.
# MAGIC - Compare trials in the [MLflow experiment](#mlflow/experiments/1645932984821329).
# MAGIC - Clone this notebook into your project folder by selecting **File > Clone** in the notebook toolbar.

# COMMAND ----------

import mlflow
import databricks.automl_runtime

target_col = "actual_increment"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

import mlflow
import os
import uuid
import shutil
import pandas as pd

# Create temp directory to download input data from MLflow
input_temp_dir = os.path.join(os.environ["SPARK_LOCAL_DIRS"], "tmp", str(uuid.uuid4())[:8])
os.makedirs(input_temp_dir)


# Download the artifact and read it into a pandas DataFrame
input_data_path = mlflow.artifacts.download_artifacts(run_id="00e03aed0b944678b90053837c425e12", artifact_path="data", dst_path=input_temp_dir)

df_loaded = pd.read_parquet(os.path.join(input_data_path, "training_data"))
# Delete the temp data
shutil.rmtree(input_temp_dir)

# Preview data
display(df_loaded.head(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Select supported columns
# MAGIC Select only the columns that are supported. This allows us to train a model that can predict on a dataset that has extra columns that are not used in training.
# MAGIC `["fib_to_centroid", "features"]` are dropped in the pipelines. See the Alerts tab of the AutoML Experiment page for details on why these columns are dropped.

# COMMAND ----------

from databricks.automl_runtime.sklearn.column_selector import ColumnSelector
supported_cols = ["summary_dim_52", "summary_dim_197", "summary_dim_227", "summary_dim_100", "desc_dim_277", "summary_dim_339", "summary_dim_291", "desc_dim_304", "desc_dim_368", "summary_dim_186", "desc_dim_197", "summary_dim_131", "summary_dim_281", "desc_dim_89", "summary_dim_166", "desc_dim_14", "desc_dim_75", "summary_dim_368", "summary_dim_21", "summary_dim_169", "desc_dim_235", "desc_dim_377", "desc_dim_252", "desc_dim_310", "desc_dim_223", "desc_dim_359", "desc_dim_60", "summary_dim_260", "desc_dim_263", "summary_dim_105", "desc_dim_233", "summary_dim_310", "summary_dim_219", "summary_dim_341", "summary_dim_361", "summary_dim_338", "desc_dim_30", "summary_dim_129", "desc_dim_349", "desc_dim_254", "summary_dim_358", "summary_dim_48", "summary_dim_26", "summary_dim_135", "desc_dim_69", "summary_dim_32", "summary_dim_13", "desc_dim_108", "desc_dim_309", "summary_dim_108", "summary_dim_53", "desc_dim_45", "desc_dim_27", "desc_dim_70", "summary_dim_238", "summary_dim_211", "desc_dim_54", "summary_dim_180", "summary_dim_184", "desc_dim_150", "resolutiondate_unix", "desc_dim_122", "summary_dim_230", "summary_dim_44", "desc_dim_1", "created_cleaned", "desc_dim_374", "centroids_padded", "desc_dim_121", "summary_dim_380", "summary_dim_175", "summary_dim_195", "desc_dim_239", "desc_dim_348", "desc_dim_217", "summary_dim_213", "summary_dim_103", "summary_dim_236", "desc_dim_251", "desc_dim_39", "summary_dim_267", "desc_dim_21", "summary_dim_161", "summary_dim_37", "summary_dim_143", "desc_dim_145", "desc_dim_232", "summary_dim_11", "desc_dim_190", "desc_dim_255", "summary_dim_144", "summary_dim_258", "desc_dim_280", "desc_dim_240", "desc_dim_182", "desc_dim_80", "desc_dim_103", "desc_dim_316", "desc_dim_92", "summary_dim_363", "desc_dim_253", "desc_dim_285", "summary_dim_188", "desc_dim_135", "desc_dim_82", "summary_dim_162", "summary_dim_86", "desc_dim_98", "summary_dim_264", "summary_dim_292", "summary_dim_72", "desc_dim_50", "summary_dim_139", "desc_dim_168", "desc_dim_237", "summary_dim_331", "desc_dim_78", "summary_dim_279", "summary_dim_98", "summary_dim_61", "summary_dim_49", "desc_dim_266", "desc_dim_90", "desc_dim_18", "summary_dim_333", "desc_dim_53", "desc_dim_42", "summary_dim_301", "summary_dim_1", "resolutiondate_cleaned", "desc_dim_153", "desc_dim_225", "desc_dim_289", "desc_dim_339", "desc_dim_161", "desc_dim_381", "desc_dim_273", "desc_dim_163", "desc_dim_62", "summary_dim_102", "summary_dim_155", "summary_dim_178", "summary_dim_289", "desc_dim_294", "desc_dim_130", "desc_dim_111", "desc_dim_244", "summary_dim_179", "summary_dim_6", "desc_dim_346", "desc_dim_321", "summary_dim_148", "summary_dim_373", "desc_dim_181", "desc_dim_247", "summary_dim_320", "summary_dim_67", "desc_dim_330", "summary_dim_218", "summary_dim_81", "desc_dim_162", "summary_dim_352", "summary_dim_204", "summary_dim_140", "desc_dim_241", "summary_dim_70", "desc_dim_292", "summary_dim_22", "summary_dim_223", "summary_dim_20", "desc_dim_144", "summary_dim_342", "summary_dim_323", "desc_dim_94", "summary_dim_210", "desc_dim_93", "summary_dim_262", "summary_dim_150", "desc_dim_342", "summary_dim_381", "summary_dim_141", "summary_dim_87", "desc_dim_22", "summary_dim_233", "summary_dim_146", "summary_dim_167", "summary_dim_322", "desc_dim_355", "summary_dim_366", "desc_dim_205", "desc_dim_51", "desc_dim_224", "summary_dim_110", "desc_dim_59", "summary_dim_43", "desc_dim_365", "desc_dim_369", "desc_dim_328", "summary_dim_298", "desc_dim_86", "summary_dim_130", "summary_dim_309", "summary_dim_128", "summary_dim_12", "summary_dim_104", "summary_dim_378", "summary_dim_382", "summary_dim_228", "desc_dim_148", "summary_dim_183", "desc_dim_186", "desc_dim_221", "summary_dim_265", "desc_dim_87", "cycle_hours", "desc_dim_295", "desc_dim_336", "desc_dim_49", "desc_dim_302", "desc_dim_84", "summary_dim_337", "desc_dim_366", "summary_dim_347", "summary_dim_116", "summary_dim_79", "desc_dim_85", "desc_dim_172", "desc_dim_242", "desc_dim_105", "summary_dim_350", "summary_dim_250", "summary_dim_147", "desc_dim_195", "created_unix", "desc_dim_8", "desc_dim_282", "summary_dim_355", "cycle_centroids", "summary_dim_349", "summary_dim_277", "summary_dim_19", "desc_dim_100", "summary_dim_168", "desc_dim_347", "desc_dim_146", "summary_dim_199", "desc_dim_156", "desc_dim_363", "summary_dim_224", "desc_dim_164", "summary_dim_257", "desc_dim_123", "desc_dim_36", "summary_dim_55", "summary_dim_66", "desc_dim_77", "desc_dim_236", "desc_dim_284", "summary_dim_106", "summary_dim_122", "summary_dim_33", "summary_dim_83", "summary_dim_208", "desc_dim_281", "desc_dim_270", "desc_dim_184", "desc_dim_305", "summary_dim_225", "summary_dim_240", "summary_dim_243", "desc_dim_272", "desc_dim_318", "summary_dim_275", "summary_dim_18", "desc_dim_165", "desc_dim_140", "desc_dim_313", "desc_dim_52", "desc_dim_246", "summary_dim_295", "summary_dim_346", "desc_dim_177", "desc_dim_264", "summary_dim_287", "desc_dim_276", "desc_dim_298", "desc_dim_364", "summary_dim_351", "summary_dim_137", "summary_dim_154", "summary_dim_40", "summary_dim_73", "summary_dim_132", "summary_dim_173", "desc_dim_67", "desc_dim_133", "summary_dim_163", "summary_dim_217", "desc_dim_279", "desc_dim_202", "desc_dim_102", "summary_dim_117", "summary_dim_54", "summary_dim_299", "summary_dim_138", "summary_dim_212", "summary_dim_305", "summary_dim_330", "summary_dim_274", "summary_dim_36", "desc_dim_33", "desc_dim_61", "summary_dim_46", "summary_dim_92", "summary_dim_304", "desc_dim_271", "desc_dim_356", "summary_dim_7", "desc_dim_72", "summary_dim_336", "desc_dim_230", "desc_dim_201", "desc_dim_300", "summary_dim_321", "summary_dim_159", "summary_dim_307", "summary_dim_383", "desc_dim_9", "summary_dim_27", "summary_dim_276", "desc_dim_74", "desc_dim_320", "desc_dim_28", "summary_dim_95", "desc_dim_128", "desc_dim_337", "desc_dim_117", "summary_dim_193", "desc_dim_358", "summary_dim_369", "summary_dim_364", "desc_dim_262", "desc_dim_137", "desc_dim_211", "summary_dim_182", "summary_dim_226", "desc_dim_372", "summary_dim_209", "summary_dim_23", "desc_dim_218", "summary_dim_91", "summary_dim_375", "summary_dim_80", "desc_dim_25", "summary_dim_189", "desc_dim_227", "desc_dim_345", "summary_dim_77", "desc_dim_352", "summary_dim_156", "summary_dim_214", "summary_dim_263", "summary_dim_14", "summary_dim_71", "summary_dim_200", "desc_dim_207", "summary_dim_319", "desc_dim_191", "desc_dim_134", "summary_dim_113", "desc_dim_138", "desc_dim_267", "desc_dim_212", "desc_dim_362", "desc_dim_357", "summary_dim_90", "desc_dim_46", "summary_dim_97", "desc_dim_187", "desc_dim_113", "summary_dim_63", "summary_dim_82", "desc_dim_88", "desc_dim_173", "desc_dim_338", "desc_dim_127", "desc_dim_171", "summary_dim_31", "desc_dim_329", "desc_dim_152", "desc_dim_228", "summary_dim_9", "summary_dim_58", "summary_dim_101", "desc_dim_319", "summary_dim_30", "desc_dim_79", "summary_dim_232", "desc_dim_63", "desc_dim_110", "summary_dim_136", "desc_dim_380", "desc_dim_43", "summary_dim_370", "desc_dim_354", "desc_dim_245", "summary_dim_247", "desc_dim_48", "summary_dim_145", "desc_dim_101", "summary_dim_78", "desc_dim_322", "desc_dim_65", "summary_dim_302", "desc_dim_155", "desc_dim_58", "desc_dim_198", "summary_dim_196", "desc_dim_216", "desc_dim_38", "desc_dim_199", "desc_dim_333", "summary_dim_221", "desc_dim_81", "desc_dim_303", "summary_dim_266", "summary_dim_343", "summary_dim_45", "desc_dim_326", "summary_dim_201", "summary_dim_294", "summary_dim_345", "summary_dim_2", "desc_dim_115", "desc_dim_204", "summary_dim_56", "desc_dim_203", "summary_dim_39", "desc_dim_159", "summary_dim_198", "desc_dim_64", "summary_dim_120", "summary_dim_151", "desc_dim_196", "summary_dim_286", "desc_dim_213", "desc_dim_16", "desc_dim_323", "summary_dim_88", "desc_dim_340", "desc_dim_341", "desc_dim_129", "desc_dim_214", "summary_dim_255", "desc_dim_350", "desc_dim_26", "desc_dim_293", "summary_dim_177", "desc_dim_360", "desc_dim_68", "desc_dim_124", "summary_dim_157", "desc_dim_375", "summary_dim_282", "desc_dim_19", "summary_dim_285", "summary_dim_317", "summary_dim_60", "summary_dim_17", "summary_dim_76", "desc_dim_343", "summary_dim_254", "summary_dim_315", "summary_dim_93", "summary_dim_165", "summary_dim_256", "desc_dim_332", "summary_dim_152", "summary_dim_187", "summary_dim_365", "summary_dim_216", "summary_dim_384", "desc_dim_10", "desc_dim_371", "desc_dim_112", "desc_dim_243", "desc_dim_73", "desc_dim_259", "summary_dim_249", "summary_dim_176", "summary_dim_300", "summary_dim_15", "desc_dim_367", "desc_dim_296", "desc_dim_180", "summary_dim_278", "summary_dim_28", "desc_dim_185", "desc_dim_13", "summary_dim_306", "summary_dim_252", "summary_dim_205", "summary_dim_234", "summary_dim_115", "summary_dim_172", "summary_dim_251", "desc_dim_250", "summary_dim_5", "desc_dim_176", "desc_dim_210", "desc_dim_34", "desc_dim_170", "desc_dim_265", "summary_dim_326", "summary_dim_222", "summary_dim_372", "summary_dim_312", "desc_dim_351", "desc_dim_96", "summary_dim_4", "desc_dim_106", "desc_dim_37", "summary_dim_24", "desc_dim_208", "summary_dim_42", "desc_dim_308", "desc_dim_40", "summary_dim_273", "desc_dim_174", "summary_dim_8", "desc_dim_194", "desc_dim_231", "desc_dim_151", "summary_dim_371", "desc_dim_283", "desc_dim_23", "desc_dim_147", "desc_dim_286", "summary_dim_126", "desc_dim_57", "summary_dim_220", "desc_dim_24", "summary_dim_231", "sorted_centroids", "summary_dim_288", "summary_dim_327", "summary_dim_94", "desc_dim_315", "summary_dim_191", "desc_dim_167", "desc_dim_118", "summary_dim_114", "summary_dim_245", "desc_dim_119", "desc_dim_35", "desc_dim_47", "cycle_centroids_rounded", "summary_dim_194", "summary_dim_348", "summary_dim_328", "desc_dim_83", "desc_dim_20", "summary_dim_367", "desc_dim_192", "summary_dim_142", "summary_dim_324", "summary_dim_123", "summary_dim_69", "summary_dim_85", "summary_dim_237", "desc_dim_11", "desc_dim_95", "desc_dim_104", "summary_dim_192", "desc_dim_269", "summary_dim_335", "desc_dim_12", "summary_dim_360", "desc_dim_275", "summary_dim_354", "summary_dim_248", "desc_dim_378", "summary_dim_229", "summary_dim_164", "desc_dim_384", "summary_dim_362", "desc_dim_376", "desc_dim_249", "desc_dim_120", "desc_dim_297", "summary_dim_174", "summary_dim_313", "summary_dim_316", "summary_dim_107", "desc_dim_361", "summary_dim_59", "summary_dim_3", "summary_dim_269", "summary_dim_62", "desc_dim_334", "summary_dim_57", "desc_dim_261", "summary_dim_99", "summary_dim_121", "summary_dim_202", "summary_dim_353", "summary_dim_47", "desc_dim_312", "desc_dim_126", "summary_dim_25", "summary_dim_84", "desc_dim_226", "summary_dim_158", "summary_dim_29", "desc_dim_301", "desc_dim_344", "summary_dim_124", "summary_dim_311", "desc_dim_15", "summary_dim_239", "desc_dim_114", "summary_dim_181", "desc_dim_209", "summary_dim_215", "desc_dim_136", "summary_dim_134", "desc_dim_222", "summary_dim_253", "desc_dim_220", "desc_dim_206", "summary_dim_334", "summary_dim_284", "summary_dim_235", "summary_dim_303", "desc_dim_188", "desc_dim_258", "desc_dim_379", "summary_dim_149", "desc_dim_183", "summary_dim_318", "desc_dim_41", "desc_dim_325", "summary_dim_51", "desc_dim_5", "desc_dim_370", "desc_dim_324", "desc_dim_353", "summary_dim_74", "desc_dim_248", "summary_dim_290", "desc_dim_31", "desc_dim_143", "desc_dim_382", "summary_dim_246", "desc_dim_256", "desc_dim_179", "summary_dim_207", "desc_dim_99", "desc_dim_257", "desc_dim_311", "desc_dim_3", "summary_dim_127", "desc_dim_331", "summary_dim_308", "desc_dim_169", "desc_dim_109", "summary_dim_96", "summary_dim_357", "desc_dim_17", "summary_dim_38", "desc_dim_141", "desc_dim_158", "desc_dim_189", "desc_dim_32", "desc_dim_219", "summary_dim_119", "summary_dim_125", "desc_dim_142", "summary_dim_133", "summary_dim_185", "summary_dim_153", "desc_dim_56", "summary_dim_271", "desc_dim_76", "desc_dim_160", "desc_dim_307", "summary_dim_376", "summary_dim_65", "desc_dim_229", "desc_dim_149", "summary_dim_377", "summary_dim_261", "desc_dim_29", "summary_dim_203", "desc_dim_7", "desc_dim_260", "summary_dim_241", "summary_dim_109", "summary_dim_296", "desc_dim_193", "desc_dim_91", "summary_dim_190", "desc_dim_215", "desc_dim_383", "summary_dim_34", "desc_dim_55", "desc_dim_154", "desc_dim_327", "summary_dim_68", "desc_dim_107", "desc_dim_166", "desc_dim_288", "summary_dim_64", "desc_dim_335", "summary_dim_359", "desc_dim_44", "summary_dim_293", "summary_dim_356", "desc_dim_116", "summary_dim_50", "desc_dim_6", "summary_dim_75", "summary_dim_41", "desc_dim_175", "summary_dim_283", "desc_dim_287", "summary_dim_344", "summary_dim_16", "desc_dim_278", "desc_dim_306", "desc_dim_291", "summary_dim_244", "desc_dim_132", "desc_dim_125", "summary_dim_329", "desc_dim_238", "summary_dim_112", "summary_dim_340", "summary_dim_111", "desc_dim_268", "summary_dim_118", "summary_dim_259", "desc_dim_4", "summary_dim_272", "desc_dim_274", "summary_dim_89", "summary_dim_297", "summary_dim_374", "summary_dim_379", "desc_dim_290", "desc_dim_373", "summary_dim_280", "desc_dim_317", "summary_dim_35", "desc_dim_200", "summary_dim_325", "desc_dim_299", "summary_dim_171", "summary_dim_332", "desc_dim_71", "summary_dim_10", "desc_dim_66", "desc_dim_139", "summary_dim_170", "desc_dim_234", "summary_dim_268", "summary_dim_270", "desc_dim_314", "summary_dim_160", "summary_dim_242", "summary_dim_314", "desc_dim_2", "desc_dim_131", "desc_dim_97", "summary_dim_206", "desc_dim_178", "desc_dim_157"]
col_selector = ColumnSelector(supported_cols)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocessors

# COMMAND ----------

# MAGIC %md
# MAGIC ### Datetime Preprocessor
# MAGIC For each datetime column, extract relevant information from the date:
# MAGIC - Unix timestamp
# MAGIC - whether the date is a weekend
# MAGIC - whether the date is a holiday
# MAGIC
# MAGIC Additionally, extract extra information from columns with timestamps:
# MAGIC - hour of the day (one-hot encoded)
# MAGIC
# MAGIC For cyclic features, plot the values along a unit circle to encode temporal proximity:
# MAGIC - hour of the day
# MAGIC - hours since the beginning of the week
# MAGIC - hours since the beginning of the month
# MAGIC - hours since the beginning of the year

# COMMAND ----------

from pandas import Timestamp
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from databricks.automl_runtime.sklearn import DatetimeImputer
from databricks.automl_runtime.sklearn import OneHotEncoder
from databricks.automl_runtime.sklearn import TimestampTransformer
from sklearn.preprocessing import StandardScaler

imputers = {
  "created_cleaned": DatetimeImputer(),
  "resolutiondate_cleaned": DatetimeImputer(),
}

datetime_transformers = []

for col in ["created_cleaned", "resolutiondate_cleaned"]:
    ohe_transformer = ColumnTransformer(
        [("ohe", OneHotEncoder(sparse=False, handle_unknown="indicator"), [TimestampTransformer.HOUR_COLUMN_INDEX])],
        remainder="passthrough")
    timestamp_preprocessor = Pipeline([
        (f"impute_{col}", imputers[col]),
        (f"transform_{col}", TimestampTransformer()),
        (f"onehot_encode_{col}", ohe_transformer),
        (f"standardize_{col}", StandardScaler()),
    ])
    datetime_transformers.append((f"timestamp_{col}", timestamp_preprocessor, [col]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Numerical columns
# MAGIC
# MAGIC Missing values for numerical columns are imputed with mean by default.

# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

num_imputers = []
num_imputers.append(("impute_mean", SimpleImputer(), ["created_unix", "cycle_hours", "desc_dim_1", "desc_dim_10", "desc_dim_100", "desc_dim_101", "desc_dim_102", "desc_dim_103", "desc_dim_104", "desc_dim_105", "desc_dim_106", "desc_dim_107", "desc_dim_108", "desc_dim_109", "desc_dim_11", "desc_dim_110", "desc_dim_111", "desc_dim_112", "desc_dim_113", "desc_dim_114", "desc_dim_115", "desc_dim_116", "desc_dim_117", "desc_dim_118", "desc_dim_119", "desc_dim_12", "desc_dim_120", "desc_dim_121", "desc_dim_122", "desc_dim_123", "desc_dim_124", "desc_dim_125", "desc_dim_126", "desc_dim_127", "desc_dim_128", "desc_dim_129", "desc_dim_13", "desc_dim_130", "desc_dim_131", "desc_dim_132", "desc_dim_133", "desc_dim_134", "desc_dim_135", "desc_dim_136", "desc_dim_137", "desc_dim_138", "desc_dim_139", "desc_dim_14", "desc_dim_140", "desc_dim_141", "desc_dim_142", "desc_dim_143", "desc_dim_144", "desc_dim_145", "desc_dim_146", "desc_dim_147", "desc_dim_148", "desc_dim_149", "desc_dim_15", "desc_dim_150", "desc_dim_151", "desc_dim_152", "desc_dim_153", "desc_dim_154", "desc_dim_155", "desc_dim_156", "desc_dim_157", "desc_dim_158", "desc_dim_159", "desc_dim_16", "desc_dim_160", "desc_dim_161", "desc_dim_162", "desc_dim_163", "desc_dim_164", "desc_dim_165", "desc_dim_166", "desc_dim_167", "desc_dim_168", "desc_dim_169", "desc_dim_17", "desc_dim_170", "desc_dim_171", "desc_dim_172", "desc_dim_173", "desc_dim_174", "desc_dim_175", "desc_dim_176", "desc_dim_177", "desc_dim_178", "desc_dim_179", "desc_dim_18", "desc_dim_180", "desc_dim_181", "desc_dim_182", "desc_dim_183", "desc_dim_184", "desc_dim_185", "desc_dim_186", "desc_dim_187", "desc_dim_188", "desc_dim_189", "desc_dim_19", "desc_dim_190", "desc_dim_191", "desc_dim_192", "desc_dim_193", "desc_dim_194", "desc_dim_195", "desc_dim_196", "desc_dim_197", "desc_dim_198", "desc_dim_199", "desc_dim_2", "desc_dim_20", "desc_dim_200", "desc_dim_201", "desc_dim_202", "desc_dim_203", "desc_dim_204", "desc_dim_205", "desc_dim_206", "desc_dim_207", "desc_dim_208", "desc_dim_209", "desc_dim_21", "desc_dim_210", "desc_dim_211", "desc_dim_212", "desc_dim_213", "desc_dim_214", "desc_dim_215", "desc_dim_216", "desc_dim_217", "desc_dim_218", "desc_dim_219", "desc_dim_22", "desc_dim_220", "desc_dim_221", "desc_dim_222", "desc_dim_223", "desc_dim_224", "desc_dim_225", "desc_dim_226", "desc_dim_227", "desc_dim_228", "desc_dim_229", "desc_dim_23", "desc_dim_230", "desc_dim_231", "desc_dim_232", "desc_dim_233", "desc_dim_234", "desc_dim_235", "desc_dim_236", "desc_dim_237", "desc_dim_238", "desc_dim_239", "desc_dim_24", "desc_dim_240", "desc_dim_241", "desc_dim_242", "desc_dim_243", "desc_dim_244", "desc_dim_245", "desc_dim_246", "desc_dim_247", "desc_dim_248", "desc_dim_249", "desc_dim_25", "desc_dim_250", "desc_dim_251", "desc_dim_252", "desc_dim_253", "desc_dim_254", "desc_dim_255", "desc_dim_256", "desc_dim_257", "desc_dim_258", "desc_dim_259", "desc_dim_26", "desc_dim_260", "desc_dim_261", "desc_dim_262", "desc_dim_263", "desc_dim_264", "desc_dim_265", "desc_dim_266", "desc_dim_267", "desc_dim_268", "desc_dim_269", "desc_dim_27", "desc_dim_270", "desc_dim_271", "desc_dim_272", "desc_dim_273", "desc_dim_274", "desc_dim_275", "desc_dim_276", "desc_dim_277", "desc_dim_278", "desc_dim_279", "desc_dim_28", "desc_dim_280", "desc_dim_281", "desc_dim_282", "desc_dim_283", "desc_dim_284", "desc_dim_285", "desc_dim_286", "desc_dim_287", "desc_dim_288", "desc_dim_289", "desc_dim_29", "desc_dim_290", "desc_dim_291", "desc_dim_292", "desc_dim_293", "desc_dim_294", "desc_dim_295", "desc_dim_296", "desc_dim_297", "desc_dim_298", "desc_dim_299", "desc_dim_3", "desc_dim_30", "desc_dim_300", "desc_dim_301", "desc_dim_302", "desc_dim_303", "desc_dim_304", "desc_dim_305", "desc_dim_306", "desc_dim_307", "desc_dim_308", "desc_dim_309", "desc_dim_31", "desc_dim_310", "desc_dim_311", "desc_dim_312", "desc_dim_313", "desc_dim_314", "desc_dim_315", "desc_dim_316", "desc_dim_317", "desc_dim_318", "desc_dim_319", "desc_dim_32", "desc_dim_320", "desc_dim_321", "desc_dim_322", "desc_dim_323", "desc_dim_324", "desc_dim_325", "desc_dim_326", "desc_dim_327", "desc_dim_328", "desc_dim_329", "desc_dim_33", "desc_dim_330", "desc_dim_331", "desc_dim_332", "desc_dim_333", "desc_dim_334", "desc_dim_335", "desc_dim_336", "desc_dim_337", "desc_dim_338", "desc_dim_339", "desc_dim_34", "desc_dim_340", "desc_dim_341", "desc_dim_342", "desc_dim_343", "desc_dim_344", "desc_dim_345", "desc_dim_346", "desc_dim_347", "desc_dim_348", "desc_dim_349", "desc_dim_35", "desc_dim_350", "desc_dim_351", "desc_dim_352", "desc_dim_353", "desc_dim_354", "desc_dim_355", "desc_dim_356", "desc_dim_357", "desc_dim_358", "desc_dim_359", "desc_dim_36", "desc_dim_360", "desc_dim_361", "desc_dim_362", "desc_dim_363", "desc_dim_364", "desc_dim_365", "desc_dim_366", "desc_dim_367", "desc_dim_368", "desc_dim_369", "desc_dim_37", "desc_dim_370", "desc_dim_371", "desc_dim_372", "desc_dim_373", "desc_dim_374", "desc_dim_375", "desc_dim_376", "desc_dim_377", "desc_dim_378", "desc_dim_379", "desc_dim_38", "desc_dim_380", "desc_dim_381", "desc_dim_382", "desc_dim_383", "desc_dim_384", "desc_dim_39", "desc_dim_4", "desc_dim_40", "desc_dim_41", "desc_dim_42", "desc_dim_43", "desc_dim_44", "desc_dim_45", "desc_dim_46", "desc_dim_47", "desc_dim_48", "desc_dim_49", "desc_dim_5", "desc_dim_50", "desc_dim_51", "desc_dim_52", "desc_dim_53", "desc_dim_54", "desc_dim_55", "desc_dim_56", "desc_dim_57", "desc_dim_58", "desc_dim_59", "desc_dim_6", "desc_dim_60", "desc_dim_61", "desc_dim_62", "desc_dim_63", "desc_dim_64", "desc_dim_65", "desc_dim_66", "desc_dim_67", "desc_dim_68", "desc_dim_69", "desc_dim_7", "desc_dim_70", "desc_dim_71", "desc_dim_72", "desc_dim_73", "desc_dim_74", "desc_dim_75", "desc_dim_76", "desc_dim_77", "desc_dim_78", "desc_dim_79", "desc_dim_8", "desc_dim_80", "desc_dim_81", "desc_dim_82", "desc_dim_83", "desc_dim_84", "desc_dim_85", "desc_dim_86", "desc_dim_87", "desc_dim_88", "desc_dim_89", "desc_dim_9", "desc_dim_90", "desc_dim_91", "desc_dim_92", "desc_dim_93", "desc_dim_94", "desc_dim_95", "desc_dim_96", "desc_dim_97", "desc_dim_98", "desc_dim_99", "resolutiondate_unix", "summary_dim_1", "summary_dim_10", "summary_dim_100", "summary_dim_101", "summary_dim_102", "summary_dim_103", "summary_dim_104", "summary_dim_105", "summary_dim_106", "summary_dim_107", "summary_dim_108", "summary_dim_109", "summary_dim_11", "summary_dim_110", "summary_dim_111", "summary_dim_112", "summary_dim_113", "summary_dim_114", "summary_dim_115", "summary_dim_116", "summary_dim_117", "summary_dim_118", "summary_dim_119", "summary_dim_12", "summary_dim_120", "summary_dim_121", "summary_dim_122", "summary_dim_123", "summary_dim_124", "summary_dim_125", "summary_dim_126", "summary_dim_127", "summary_dim_128", "summary_dim_129", "summary_dim_13", "summary_dim_130", "summary_dim_131", "summary_dim_132", "summary_dim_133", "summary_dim_134", "summary_dim_135", "summary_dim_136", "summary_dim_137", "summary_dim_138", "summary_dim_139", "summary_dim_14", "summary_dim_140", "summary_dim_141", "summary_dim_142", "summary_dim_143", "summary_dim_144", "summary_dim_145", "summary_dim_146", "summary_dim_147", "summary_dim_148", "summary_dim_149", "summary_dim_15", "summary_dim_150", "summary_dim_151", "summary_dim_152", "summary_dim_153", "summary_dim_154", "summary_dim_155", "summary_dim_156", "summary_dim_157", "summary_dim_158", "summary_dim_159", "summary_dim_16", "summary_dim_160", "summary_dim_161", "summary_dim_162", "summary_dim_163", "summary_dim_164", "summary_dim_165", "summary_dim_166", "summary_dim_167", "summary_dim_168", "summary_dim_169", "summary_dim_17", "summary_dim_170", "summary_dim_171", "summary_dim_172", "summary_dim_173", "summary_dim_174", "summary_dim_175", "summary_dim_176", "summary_dim_177", "summary_dim_178", "summary_dim_179", "summary_dim_18", "summary_dim_180", "summary_dim_181", "summary_dim_182", "summary_dim_183", "summary_dim_184", "summary_dim_185", "summary_dim_186", "summary_dim_187", "summary_dim_188", "summary_dim_189", "summary_dim_19", "summary_dim_190", "summary_dim_191", "summary_dim_192", "summary_dim_193", "summary_dim_194", "summary_dim_195", "summary_dim_196", "summary_dim_197", "summary_dim_198", "summary_dim_199", "summary_dim_2", "summary_dim_20", "summary_dim_200", "summary_dim_201", "summary_dim_202", "summary_dim_203", "summary_dim_204", "summary_dim_205", "summary_dim_206", "summary_dim_207", "summary_dim_208", "summary_dim_209", "summary_dim_21", "summary_dim_210", "summary_dim_211", "summary_dim_212", "summary_dim_213", "summary_dim_214", "summary_dim_215", "summary_dim_216", "summary_dim_217", "summary_dim_218", "summary_dim_219", "summary_dim_22", "summary_dim_220", "summary_dim_221", "summary_dim_222", "summary_dim_223", "summary_dim_224", "summary_dim_225", "summary_dim_226", "summary_dim_227", "summary_dim_228", "summary_dim_229", "summary_dim_23", "summary_dim_230", "summary_dim_231", "summary_dim_232", "summary_dim_233", "summary_dim_234", "summary_dim_235", "summary_dim_236", "summary_dim_237", "summary_dim_238", "summary_dim_239", "summary_dim_24", "summary_dim_240", "summary_dim_241", "summary_dim_242", "summary_dim_243", "summary_dim_244", "summary_dim_245", "summary_dim_246", "summary_dim_247", "summary_dim_248", "summary_dim_249", "summary_dim_25", "summary_dim_250", "summary_dim_251", "summary_dim_252", "summary_dim_253", "summary_dim_254", "summary_dim_255", "summary_dim_256", "summary_dim_257", "summary_dim_258", "summary_dim_259", "summary_dim_26", "summary_dim_260", "summary_dim_261", "summary_dim_262", "summary_dim_263", "summary_dim_264", "summary_dim_265", "summary_dim_266", "summary_dim_267", "summary_dim_268", "summary_dim_269", "summary_dim_27", "summary_dim_270", "summary_dim_271", "summary_dim_272", "summary_dim_273", "summary_dim_274", "summary_dim_275", "summary_dim_276", "summary_dim_277", "summary_dim_278", "summary_dim_279", "summary_dim_28", "summary_dim_280", "summary_dim_281", "summary_dim_282", "summary_dim_283", "summary_dim_284", "summary_dim_285", "summary_dim_286", "summary_dim_287", "summary_dim_288", "summary_dim_289", "summary_dim_29", "summary_dim_290", "summary_dim_291", "summary_dim_292", "summary_dim_293", "summary_dim_294", "summary_dim_295", "summary_dim_296", "summary_dim_297", "summary_dim_298", "summary_dim_299", "summary_dim_3", "summary_dim_30", "summary_dim_300", "summary_dim_301", "summary_dim_302", "summary_dim_303", "summary_dim_304", "summary_dim_305", "summary_dim_306", "summary_dim_307", "summary_dim_308", "summary_dim_309", "summary_dim_31", "summary_dim_310", "summary_dim_311", "summary_dim_312", "summary_dim_313", "summary_dim_314", "summary_dim_315", "summary_dim_316", "summary_dim_317", "summary_dim_318", "summary_dim_319", "summary_dim_32", "summary_dim_320", "summary_dim_321", "summary_dim_322", "summary_dim_323", "summary_dim_324", "summary_dim_325", "summary_dim_326", "summary_dim_327", "summary_dim_328", "summary_dim_329", "summary_dim_33", "summary_dim_330", "summary_dim_331", "summary_dim_332", "summary_dim_333", "summary_dim_334", "summary_dim_335", "summary_dim_336", "summary_dim_337", "summary_dim_338", "summary_dim_339", "summary_dim_34", "summary_dim_340", "summary_dim_341", "summary_dim_342", "summary_dim_343", "summary_dim_344", "summary_dim_345", "summary_dim_346", "summary_dim_347", "summary_dim_348", "summary_dim_349", "summary_dim_35", "summary_dim_350", "summary_dim_351", "summary_dim_352", "summary_dim_353", "summary_dim_354", "summary_dim_355", "summary_dim_356", "summary_dim_357", "summary_dim_358", "summary_dim_359", "summary_dim_36", "summary_dim_360", "summary_dim_361", "summary_dim_362", "summary_dim_363", "summary_dim_364", "summary_dim_365", "summary_dim_366", "summary_dim_367", "summary_dim_368", "summary_dim_369", "summary_dim_37", "summary_dim_370", "summary_dim_371", "summary_dim_372", "summary_dim_373", "summary_dim_374", "summary_dim_375", "summary_dim_376", "summary_dim_377", "summary_dim_378", "summary_dim_379", "summary_dim_38", "summary_dim_380", "summary_dim_381", "summary_dim_382", "summary_dim_383", "summary_dim_384", "summary_dim_39", "summary_dim_4", "summary_dim_40", "summary_dim_41", "summary_dim_42", "summary_dim_43", "summary_dim_44", "summary_dim_45", "summary_dim_46", "summary_dim_47", "summary_dim_48", "summary_dim_49", "summary_dim_5", "summary_dim_50", "summary_dim_51", "summary_dim_52", "summary_dim_53", "summary_dim_54", "summary_dim_55", "summary_dim_56", "summary_dim_57", "summary_dim_58", "summary_dim_59", "summary_dim_6", "summary_dim_60", "summary_dim_61", "summary_dim_62", "summary_dim_63", "summary_dim_64", "summary_dim_65", "summary_dim_66", "summary_dim_67", "summary_dim_68", "summary_dim_69", "summary_dim_7", "summary_dim_70", "summary_dim_71", "summary_dim_72", "summary_dim_73", "summary_dim_74", "summary_dim_75", "summary_dim_76", "summary_dim_77", "summary_dim_78", "summary_dim_79", "summary_dim_8", "summary_dim_80", "summary_dim_81", "summary_dim_82", "summary_dim_83", "summary_dim_84", "summary_dim_85", "summary_dim_86", "summary_dim_87", "summary_dim_88", "summary_dim_89", "summary_dim_9", "summary_dim_90", "summary_dim_91", "summary_dim_92", "summary_dim_93", "summary_dim_94", "summary_dim_95", "summary_dim_96", "summary_dim_97", "summary_dim_98", "summary_dim_99"]))

numerical_pipeline = Pipeline(steps=[
    ("converter", FunctionTransformer(lambda df: df.apply(pd.to_numeric, errors='coerce'))),
    ("imputers", ColumnTransformer(num_imputers)),
    ("standardizer", StandardScaler()),
])

numerical_transformers = [("numerical", numerical_pipeline, ["summary_dim_52", "summary_dim_197", "summary_dim_227", "summary_dim_100", "desc_dim_277", "summary_dim_339", "summary_dim_291", "desc_dim_304", "desc_dim_368", "summary_dim_186", "desc_dim_197", "summary_dim_131", "summary_dim_281", "desc_dim_89", "summary_dim_166", "desc_dim_14", "desc_dim_75", "summary_dim_368", "summary_dim_21", "summary_dim_169", "desc_dim_235", "desc_dim_377", "desc_dim_252", "desc_dim_310", "desc_dim_223", "desc_dim_359", "desc_dim_60", "summary_dim_260", "desc_dim_263", "summary_dim_105", "desc_dim_233", "summary_dim_310", "summary_dim_219", "summary_dim_341", "summary_dim_361", "summary_dim_338", "desc_dim_30", "summary_dim_129", "desc_dim_349", "desc_dim_254", "summary_dim_358", "summary_dim_48", "summary_dim_26", "summary_dim_135", "desc_dim_69", "summary_dim_32", "summary_dim_13", "desc_dim_108", "desc_dim_309", "summary_dim_108", "summary_dim_53", "desc_dim_45", "desc_dim_27", "desc_dim_70", "summary_dim_238", "summary_dim_211", "desc_dim_54", "summary_dim_180", "summary_dim_184", "desc_dim_150", "resolutiondate_unix", "desc_dim_122", "summary_dim_230", "summary_dim_44", "desc_dim_1", "desc_dim_374", "desc_dim_121", "summary_dim_380", "summary_dim_175", "summary_dim_195", "desc_dim_239", "desc_dim_348", "desc_dim_217", "summary_dim_213", "summary_dim_103", "summary_dim_236", "desc_dim_251", "desc_dim_39", "summary_dim_267", "desc_dim_21", "summary_dim_161", "summary_dim_37", "summary_dim_143", "desc_dim_145", "desc_dim_232", "summary_dim_11", "desc_dim_190", "desc_dim_255", "summary_dim_144", "summary_dim_258", "desc_dim_280", "desc_dim_240", "desc_dim_182", "desc_dim_80", "desc_dim_103", "desc_dim_316", "desc_dim_92", "summary_dim_363", "desc_dim_253", "desc_dim_285", "summary_dim_188", "desc_dim_135", "desc_dim_82", "summary_dim_162", "summary_dim_86", "desc_dim_98", "summary_dim_264", "summary_dim_292", "summary_dim_72", "desc_dim_50", "summary_dim_139", "desc_dim_168", "desc_dim_237", "summary_dim_331", "desc_dim_78", "summary_dim_279", "summary_dim_98", "summary_dim_61", "summary_dim_49", "desc_dim_266", "desc_dim_90", "desc_dim_18", "summary_dim_333", "desc_dim_53", "desc_dim_42", "summary_dim_301", "summary_dim_1", "desc_dim_153", "desc_dim_225", "desc_dim_289", "desc_dim_339", "desc_dim_161", "desc_dim_381", "desc_dim_273", "desc_dim_163", "desc_dim_62", "summary_dim_102", "summary_dim_155", "summary_dim_178", "summary_dim_289", "desc_dim_294", "desc_dim_130", "desc_dim_111", "desc_dim_244", "summary_dim_179", "summary_dim_6", "desc_dim_346", "desc_dim_321", "summary_dim_148", "summary_dim_373", "desc_dim_181", "desc_dim_247", "summary_dim_320", "summary_dim_67", "desc_dim_330", "summary_dim_218", "summary_dim_81", "desc_dim_162", "summary_dim_352", "summary_dim_204", "summary_dim_140", "desc_dim_241", "summary_dim_70", "desc_dim_292", "summary_dim_22", "summary_dim_223", "summary_dim_20", "desc_dim_144", "summary_dim_342", "summary_dim_323", "desc_dim_94", "summary_dim_210", "desc_dim_93", "summary_dim_262", "summary_dim_150", "desc_dim_342", "summary_dim_381", "summary_dim_141", "summary_dim_87", "desc_dim_22", "summary_dim_233", "summary_dim_146", "summary_dim_167", "summary_dim_322", "desc_dim_355", "summary_dim_366", "desc_dim_205", "desc_dim_51", "desc_dim_224", "summary_dim_110", "desc_dim_59", "summary_dim_43", "desc_dim_365", "desc_dim_369", "desc_dim_328", "summary_dim_298", "desc_dim_86", "summary_dim_130", "summary_dim_309", "summary_dim_128", "summary_dim_12", "summary_dim_104", "summary_dim_378", "summary_dim_382", "summary_dim_228", "desc_dim_148", "summary_dim_183", "desc_dim_186", "desc_dim_221", "summary_dim_265", "desc_dim_87", "cycle_hours", "desc_dim_295", "desc_dim_336", "desc_dim_49", "desc_dim_302", "desc_dim_84", "summary_dim_337", "desc_dim_366", "summary_dim_347", "summary_dim_116", "summary_dim_79", "desc_dim_85", "desc_dim_172", "desc_dim_242", "desc_dim_105", "summary_dim_350", "summary_dim_250", "summary_dim_147", "desc_dim_195", "created_unix", "desc_dim_8", "desc_dim_282", "summary_dim_355", "summary_dim_349", "summary_dim_277", "summary_dim_19", "desc_dim_100", "summary_dim_168", "desc_dim_347", "desc_dim_146", "summary_dim_199", "desc_dim_156", "desc_dim_363", "summary_dim_224", "desc_dim_164", "summary_dim_257", "desc_dim_123", "desc_dim_36", "summary_dim_55", "summary_dim_66", "desc_dim_77", "desc_dim_236", "desc_dim_284", "summary_dim_106", "summary_dim_122", "summary_dim_33", "summary_dim_83", "summary_dim_208", "desc_dim_281", "desc_dim_270", "desc_dim_184", "desc_dim_305", "summary_dim_225", "summary_dim_240", "summary_dim_243", "desc_dim_272", "desc_dim_318", "summary_dim_275", "summary_dim_18", "desc_dim_165", "desc_dim_140", "desc_dim_313", "desc_dim_52", "desc_dim_246", "summary_dim_295", "summary_dim_346", "desc_dim_177", "desc_dim_264", "summary_dim_287", "desc_dim_276", "desc_dim_298", "desc_dim_364", "summary_dim_351", "summary_dim_137", "summary_dim_154", "summary_dim_40", "summary_dim_73", "summary_dim_132", "summary_dim_173", "desc_dim_67", "desc_dim_133", "summary_dim_163", "summary_dim_217", "desc_dim_279", "desc_dim_202", "desc_dim_102", "summary_dim_117", "summary_dim_54", "summary_dim_299", "summary_dim_138", "summary_dim_212", "summary_dim_305", "summary_dim_330", "summary_dim_274", "summary_dim_36", "desc_dim_33", "desc_dim_61", "summary_dim_46", "summary_dim_92", "summary_dim_304", "desc_dim_271", "desc_dim_356", "summary_dim_7", "desc_dim_72", "summary_dim_336", "desc_dim_230", "desc_dim_201", "desc_dim_300", "summary_dim_321", "summary_dim_159", "summary_dim_307", "summary_dim_383", "desc_dim_9", "summary_dim_27", "summary_dim_276", "desc_dim_74", "desc_dim_320", "desc_dim_28", "summary_dim_95", "desc_dim_128", "desc_dim_337", "desc_dim_117", "summary_dim_193", "desc_dim_358", "summary_dim_369", "summary_dim_364", "desc_dim_262", "desc_dim_137", "desc_dim_211", "summary_dim_182", "summary_dim_226", "desc_dim_372", "summary_dim_209", "summary_dim_23", "desc_dim_218", "summary_dim_91", "summary_dim_375", "summary_dim_80", "desc_dim_25", "summary_dim_189", "desc_dim_227", "desc_dim_345", "summary_dim_77", "desc_dim_352", "summary_dim_156", "summary_dim_214", "summary_dim_263", "summary_dim_14", "summary_dim_71", "summary_dim_200", "desc_dim_207", "summary_dim_319", "desc_dim_191", "desc_dim_134", "summary_dim_113", "desc_dim_138", "desc_dim_267", "desc_dim_212", "desc_dim_362", "desc_dim_357", "summary_dim_90", "desc_dim_46", "summary_dim_97", "desc_dim_187", "desc_dim_113", "summary_dim_63", "summary_dim_82", "desc_dim_88", "desc_dim_173", "desc_dim_338", "desc_dim_127", "desc_dim_171", "summary_dim_31", "desc_dim_329", "desc_dim_152", "desc_dim_228", "summary_dim_9", "summary_dim_58", "summary_dim_101", "desc_dim_319", "summary_dim_30", "desc_dim_79", "summary_dim_232", "desc_dim_63", "desc_dim_110", "summary_dim_136", "desc_dim_380", "desc_dim_43", "summary_dim_370", "desc_dim_354", "desc_dim_245", "summary_dim_247", "desc_dim_48", "summary_dim_145", "desc_dim_101", "summary_dim_78", "desc_dim_322", "desc_dim_65", "summary_dim_302", "desc_dim_155", "desc_dim_58", "desc_dim_198", "summary_dim_196", "desc_dim_216", "desc_dim_38", "desc_dim_199", "desc_dim_333", "summary_dim_221", "desc_dim_81", "desc_dim_303", "summary_dim_266", "summary_dim_343", "summary_dim_45", "desc_dim_326", "summary_dim_201", "summary_dim_294", "summary_dim_345", "summary_dim_2", "desc_dim_115", "desc_dim_204", "summary_dim_56", "desc_dim_203", "summary_dim_39", "desc_dim_159", "summary_dim_198", "desc_dim_64", "summary_dim_120", "summary_dim_151", "desc_dim_196", "summary_dim_286", "desc_dim_213", "desc_dim_16", "desc_dim_323", "summary_dim_88", "desc_dim_340", "desc_dim_341", "desc_dim_129", "desc_dim_214", "summary_dim_255", "desc_dim_350", "desc_dim_26", "desc_dim_293", "summary_dim_177", "desc_dim_360", "desc_dim_68", "desc_dim_124", "summary_dim_157", "desc_dim_375", "summary_dim_282", "desc_dim_19", "summary_dim_285", "summary_dim_317", "summary_dim_60", "summary_dim_17", "summary_dim_76", "desc_dim_343", "summary_dim_254", "summary_dim_315", "summary_dim_93", "summary_dim_165", "summary_dim_256", "desc_dim_332", "summary_dim_152", "summary_dim_187", "summary_dim_365", "summary_dim_216", "summary_dim_384", "desc_dim_10", "desc_dim_371", "desc_dim_112", "desc_dim_243", "desc_dim_73", "desc_dim_259", "summary_dim_249", "summary_dim_176", "summary_dim_300", "summary_dim_15", "desc_dim_367", "desc_dim_296", "desc_dim_180", "summary_dim_278", "summary_dim_28", "desc_dim_185", "desc_dim_13", "summary_dim_306", "summary_dim_252", "summary_dim_205", "summary_dim_234", "summary_dim_115", "summary_dim_172", "summary_dim_251", "desc_dim_250", "summary_dim_5", "desc_dim_176", "desc_dim_210", "desc_dim_34", "desc_dim_170", "desc_dim_265", "summary_dim_326", "summary_dim_222", "summary_dim_372", "summary_dim_312", "desc_dim_351", "desc_dim_96", "summary_dim_4", "desc_dim_106", "desc_dim_37", "summary_dim_24", "desc_dim_208", "summary_dim_42", "desc_dim_308", "desc_dim_40", "summary_dim_273", "desc_dim_174", "summary_dim_8", "desc_dim_194", "desc_dim_231", "desc_dim_151", "summary_dim_371", "desc_dim_283", "desc_dim_23", "desc_dim_147", "desc_dim_286", "summary_dim_126", "desc_dim_57", "summary_dim_220", "desc_dim_24", "summary_dim_231", "summary_dim_288", "summary_dim_327", "summary_dim_94", "desc_dim_315", "summary_dim_191", "desc_dim_167", "desc_dim_118", "summary_dim_114", "summary_dim_245", "desc_dim_119", "desc_dim_35", "desc_dim_47", "summary_dim_194", "summary_dim_348", "summary_dim_328", "desc_dim_83", "desc_dim_20", "summary_dim_367", "desc_dim_192", "summary_dim_142", "summary_dim_324", "summary_dim_123", "summary_dim_69", "summary_dim_85", "summary_dim_237", "desc_dim_11", "desc_dim_95", "desc_dim_104", "summary_dim_192", "desc_dim_269", "summary_dim_335", "desc_dim_12", "summary_dim_360", "desc_dim_275", "summary_dim_354", "summary_dim_248", "desc_dim_378", "summary_dim_229", "summary_dim_164", "desc_dim_384", "summary_dim_362", "desc_dim_376", "desc_dim_249", "desc_dim_120", "desc_dim_297", "summary_dim_174", "summary_dim_313", "summary_dim_316", "summary_dim_107", "desc_dim_361", "summary_dim_59", "summary_dim_3", "summary_dim_269", "summary_dim_62", "desc_dim_334", "summary_dim_57", "desc_dim_261", "summary_dim_99", "summary_dim_121", "summary_dim_202", "summary_dim_353", "summary_dim_47", "desc_dim_312", "desc_dim_126", "summary_dim_25", "summary_dim_84", "desc_dim_226", "summary_dim_158", "summary_dim_29", "desc_dim_301", "desc_dim_344", "summary_dim_124", "summary_dim_311", "desc_dim_15", "summary_dim_239", "desc_dim_114", "summary_dim_181", "desc_dim_209", "summary_dim_215", "desc_dim_136", "summary_dim_134", "desc_dim_222", "summary_dim_253", "desc_dim_220", "desc_dim_206", "summary_dim_334", "summary_dim_284", "summary_dim_235", "summary_dim_303", "desc_dim_188", "desc_dim_258", "desc_dim_379", "summary_dim_149", "desc_dim_183", "summary_dim_318", "desc_dim_41", "desc_dim_325", "summary_dim_51", "desc_dim_5", "desc_dim_370", "desc_dim_324", "desc_dim_353", "summary_dim_74", "desc_dim_248", "summary_dim_290", "desc_dim_31", "desc_dim_143", "desc_dim_382", "summary_dim_246", "desc_dim_256", "desc_dim_179", "summary_dim_207", "desc_dim_99", "desc_dim_257", "desc_dim_311", "desc_dim_3", "summary_dim_127", "desc_dim_331", "summary_dim_308", "desc_dim_169", "desc_dim_109", "summary_dim_96", "summary_dim_357", "desc_dim_17", "summary_dim_38", "desc_dim_141", "desc_dim_158", "desc_dim_189", "desc_dim_32", "desc_dim_219", "summary_dim_119", "summary_dim_125", "desc_dim_142", "summary_dim_133", "summary_dim_185", "summary_dim_153", "desc_dim_56", "summary_dim_271", "desc_dim_76", "desc_dim_160", "desc_dim_307", "summary_dim_376", "summary_dim_65", "desc_dim_229", "desc_dim_149", "summary_dim_377", "summary_dim_261", "desc_dim_29", "summary_dim_203", "desc_dim_7", "desc_dim_260", "summary_dim_241", "summary_dim_109", "summary_dim_296", "desc_dim_193", "desc_dim_91", "summary_dim_190", "desc_dim_215", "desc_dim_383", "summary_dim_34", "desc_dim_55", "desc_dim_154", "desc_dim_327", "summary_dim_68", "desc_dim_107", "desc_dim_166", "desc_dim_288", "summary_dim_64", "desc_dim_335", "summary_dim_359", "desc_dim_44", "summary_dim_293", "summary_dim_356", "desc_dim_116", "summary_dim_50", "desc_dim_6", "summary_dim_75", "summary_dim_41", "desc_dim_175", "summary_dim_283", "desc_dim_287", "summary_dim_344", "summary_dim_16", "desc_dim_278", "desc_dim_306", "desc_dim_291", "summary_dim_244", "desc_dim_132", "desc_dim_125", "summary_dim_329", "desc_dim_238", "summary_dim_112", "summary_dim_340", "summary_dim_111", "desc_dim_268", "summary_dim_118", "summary_dim_259", "desc_dim_4", "summary_dim_272", "desc_dim_274", "summary_dim_89", "summary_dim_297", "summary_dim_374", "summary_dim_379", "desc_dim_290", "desc_dim_373", "summary_dim_280", "desc_dim_317", "summary_dim_35", "desc_dim_200", "summary_dim_325", "desc_dim_299", "summary_dim_171", "summary_dim_332", "desc_dim_71", "summary_dim_10", "desc_dim_66", "desc_dim_139", "summary_dim_170", "desc_dim_234", "summary_dim_268", "summary_dim_270", "desc_dim_314", "summary_dim_160", "summary_dim_242", "summary_dim_314", "desc_dim_2", "desc_dim_131", "desc_dim_97", "summary_dim_206", "desc_dim_178", "desc_dim_157"])]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Array columns

# COMMAND ----------

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

def concat_arrays(df):
    """Concatenate all the array columns (if more than one) into a single numpy array."""
    return np.array([np.concatenate(row) for row in df.values])


array_pipeline = Pipeline(steps=[
    ("concat", FunctionTransformer(concat_arrays)),
    ("standardize", StandardScaler()),
])

array_transformers = [["array", array_pipeline, ["centroids_padded", "cycle_centroids_rounded", "cycle_centroids", "sorted_centroids"]]]

# COMMAND ----------

from sklearn.compose import ColumnTransformer

transformers = datetime_transformers + numerical_transformers + array_transformers

preprocessor = ColumnTransformer(transformers, remainder="passthrough", sparse_threshold=0)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train - Validation - Test Split
# MAGIC The input data is split by AutoML into 3 sets:
# MAGIC - Train (60% of the dataset used to train the model)
# MAGIC - Validation (20% of the dataset used to tune the hyperparameters of the model)
# MAGIC - Test (20% of the dataset used to report the true performance of the model on an unseen dataset)
# MAGIC
# MAGIC `_automl_split_col_0000` contains the information of which set a given row belongs to.
# MAGIC We use this column to split the dataset into the above 3 sets. 
# MAGIC The column should not be used for training so it is dropped after split is done.

# COMMAND ----------

# AutoML completed train - validation - test split internally and used _automl_split_col_0000 to specify the set
split_train_df = df_loaded.loc[df_loaded._automl_split_col_0000 == "train"]
split_val_df = df_loaded.loc[df_loaded._automl_split_col_0000 == "validate"]
split_test_df = df_loaded.loc[df_loaded._automl_split_col_0000 == "test"]

# Separate target column from features and drop _automl_split_col_0000
X_train = split_train_df.drop([target_col, "_automl_split_col_0000"], axis=1)
y_train = split_train_df[target_col]

X_val = split_val_df.drop([target_col, "_automl_split_col_0000"], axis=1)
y_val = split_val_df[target_col]

X_test = split_test_df.drop([target_col, "_automl_split_col_0000"], axis=1)
y_test = split_test_df[target_col]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train regression model
# MAGIC - Log relevant metrics to MLflow to track runs
# MAGIC - All the runs are logged under [this MLflow experiment](#mlflow/experiments/1645932984821329)
# MAGIC - Change the model parameters and re-run the training cell to log a different trial to the MLflow experiment
# MAGIC - To view the full list of tunable hyperparameters, check the output of the cell below

# COMMAND ----------

from xgboost import XGBRegressor

help(XGBRegressor)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define the objective function
# MAGIC The objective function used to find optimal hyperparameters. By default, this notebook only runs
# MAGIC this function once (`max_evals=1` in the `hyperopt.fmin` invocation) with fixed hyperparameters, but
# MAGIC hyperparameters can be tuned by modifying `space`, defined below. `hyperopt.fmin` will then use this
# MAGIC function's return value to search the space to minimize the loss.

# COMMAND ----------

import mlflow
from mlflow.models import Model, infer_signature, ModelSignature
from mlflow.pyfunc import PyFuncModel
from mlflow import pyfunc
import sklearn
from sklearn import set_config
from sklearn.pipeline import Pipeline
from hyperopt import hp, tpe, fmin, STATUS_OK, Trials


# Create a separate pipeline to transform the validation dataset. This is used for early stopping.
pipeline_val = Pipeline([
    ("column_selector", col_selector),
    ("preprocessor", preprocessor),
])

mlflow.sklearn.autolog(disable=True)
pipeline_val.fit(X_train, y_train)
X_val_processed = pipeline_val.transform(X_val)

def objective(params):
  with mlflow.start_run(experiment_id="1645932984821329") as mlflow_run:
    xgb_regressor = XGBRegressor(**params)

    model = Pipeline([
        ("column_selector", col_selector),
        ("preprocessor", preprocessor),
        ("regressor", xgb_regressor),
    ])

    # Enable automatic logging of input samples, metrics, parameters, and models
    mlflow.sklearn.autolog(
        log_input_examples=True,
        silent=True,
    )

    model.fit(X_train, y_train, regressor__early_stopping_rounds=5, regressor__verbose=False, regressor__eval_set=[(X_val_processed,y_val)])

    
    # Log metrics for the training set
    mlflow_model = Model()
    pyfunc.add_to_model(mlflow_model, loader_module="mlflow.sklearn")
    pyfunc_model = PyFuncModel(model_meta=mlflow_model, model_impl=model)
    training_eval_result = mlflow.evaluate(
        model=pyfunc_model,
        data=X_train.assign(**{str(target_col):y_train}),
        targets=target_col,
        model_type="regressor",
        evaluator_config = {"log_model_explainability": False,
                            "metric_prefix": "training_"  }
    )
    # Log metrics for the validation set
    val_eval_result = mlflow.evaluate(
        model=pyfunc_model,
        data=X_val.assign(**{str(target_col):y_val}),
        targets=target_col,
        model_type="regressor",
        evaluator_config= {"log_model_explainability": False,
                           "metric_prefix": "val_"  }
   )
    xgb_val_metrics = val_eval_result.metrics
    # Log metrics for the test set
    test_eval_result = mlflow.evaluate(
        model=pyfunc_model,
        data=X_test.assign(**{str(target_col):y_test}),
        targets=target_col,
        model_type="regressor",
        evaluator_config= {"log_model_explainability": False,
                           "metric_prefix": "test_"  }
   )
    xgb_test_metrics = test_eval_result.metrics

    loss = -xgb_val_metrics["val_r2_score"]

    # Truncate metric key names so they can be displayed together
    xgb_val_metrics = {k.replace("val_", ""): v for k, v in xgb_val_metrics.items()}
    xgb_test_metrics = {k.replace("test_", ""): v for k, v in xgb_test_metrics.items()}

    return {
      "loss": loss,
      "status": STATUS_OK,
      "val_metrics": xgb_val_metrics,
      "test_metrics": xgb_test_metrics,
      "model": model,
      "run": mlflow_run,
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configure the hyperparameter search space
# MAGIC Configure the search space of parameters. Parameters below are all constant expressions but can be
# MAGIC modified to widen the search space. For example, when training a decision tree regressor, to allow
# MAGIC the maximum tree depth to be either 2 or 3, set the key of 'max_depth' to
# MAGIC `hp.choice('max_depth', [2, 3])`. Be sure to also increase `max_evals` in the `fmin` call below.
# MAGIC
# MAGIC See https://docs.databricks.com/applications/machine-learning/automl-hyperparam-tuning/index.html
# MAGIC for more information on hyperparameter tuning as well as
# MAGIC http://hyperopt.github.io/hyperopt/getting-started/search_spaces/ for documentation on supported
# MAGIC search expressions.
# MAGIC
# MAGIC For documentation on parameters used by the model in use, please see:
# MAGIC https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRegressor
# MAGIC
# MAGIC NOTE: The above URL points to a stable version of the documentation corresponding to the last
# MAGIC released version of the package. The documentation may differ slightly for the package version
# MAGIC used by this notebook.

# COMMAND ----------

space = {
  "colsample_bytree": 0.7525593567558386,
  "learning_rate": 0.029881517809014744,
  "max_depth": 11,
  "min_child_weight": 6,
  "n_estimators": 268,
  "n_jobs": 100,
  "subsample": 0.7341872201743822,
  "verbosity": 0,
  "random_state": 651849752,
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run trials
# MAGIC When widening the search space and training multiple models, switch to `SparkTrials` to parallelize
# MAGIC training on Spark:
# MAGIC ```
# MAGIC from hyperopt import SparkTrials
# MAGIC trials = SparkTrials()
# MAGIC ```
# MAGIC
# MAGIC NOTE: While `Trials` starts an MLFlow run for each set of hyperparameters, `SparkTrials` only starts
# MAGIC one top-level run; it will start a subrun for each set of hyperparameters.
# MAGIC
# MAGIC See http://hyperopt.github.io/hyperopt/scaleout/spark/ for more info.

# COMMAND ----------

trials = Trials()
fmin(objective,
     space=space,
     algo=tpe.suggest,
     max_evals=1,  # Increase this when widening the hyperparameter search space.
     trials=trials)

best_result = trials.best_trial["result"]
model = best_result["model"]
mlflow_run = best_result["run"]

display(
  pd.DataFrame(
    [best_result["val_metrics"], best_result["test_metrics"]],
    index=pd.Index(["validation", "test"], name="split")).reset_index())

set_config(display="diagram")
model

# COMMAND ----------

# MAGIC %md
# MAGIC ### Patch pandas version in logged model
# MAGIC
# MAGIC Ensures that model serving uses the same version of pandas that was used to train the model.

# COMMAND ----------

import mlflow
import os
import shutil
import tempfile
import yaml

run_id = mlflow_run.info.run_id

# Set up a local dir for downloading the artifacts.
tmp_dir = tempfile.mkdtemp()

client = mlflow.tracking.MlflowClient()

# Fix conda.yaml
conda_file_path = mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/model/conda.yaml", dst_path=tmp_dir)
with open(conda_file_path) as f:
  conda_libs = yaml.load(f, Loader=yaml.FullLoader)
pandas_lib_exists = any([lib.startswith("pandas==") for lib in conda_libs["dependencies"][-1]["pip"]])
if not pandas_lib_exists:
  print("Adding pandas dependency to conda.yaml")
  conda_libs["dependencies"][-1]["pip"].append(f"pandas=={pd.__version__}")

  with open(f"{tmp_dir}/conda.yaml", "w") as f:
    f.write(yaml.dump(conda_libs))
  client.log_artifact(run_id=run_id, local_path=conda_file_path, artifact_path="model")

# Fix requirements.txt
venv_file_path = mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/model/requirements.txt", dst_path=tmp_dir)
with open(venv_file_path) as f:
  venv_libs = f.readlines()
venv_libs = [lib.strip() for lib in venv_libs]
pandas_lib_exists = any([lib.startswith("pandas==") for lib in venv_libs])
if not pandas_lib_exists:
  print("Adding pandas dependency to requirements.txt")
  venv_libs.append(f"pandas=={pd.__version__}")

  with open(f"{tmp_dir}/requirements.txt", "w") as f:
    f.write("\n".join(venv_libs))
  client.log_artifact(run_id=run_id, local_path=venv_file_path, artifact_path="model")

shutil.rmtree(tmp_dir)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature importance
# MAGIC
# MAGIC SHAP is a game-theoretic approach to explain machine learning models, providing a summary plot
# MAGIC of the relationship between features and model output. Features are ranked in descending order of
# MAGIC importance, and impact/color describe the correlation between the feature and the target variable.
# MAGIC - Generating SHAP feature importance is a very memory intensive operation, so to ensure that AutoML can run trials without
# MAGIC   running out of memory, we disable SHAP by default.<br />
# MAGIC   You can set the flag defined below to `shap_enabled = True` and re-run this notebook to see the SHAP plots.
# MAGIC - To reduce the computational overhead of each trial, a single example is sampled from the validation set to explain.<br />
# MAGIC   For more thorough results, increase the sample size of explanations, or provide your own examples to explain.
# MAGIC - SHAP cannot explain models using data with nulls; if your dataset has any, both the background data and
# MAGIC   examples to explain will be imputed using the mode (most frequent values). This affects the computed
# MAGIC   SHAP values, as the imputed samples may not match the actual data distribution.
# MAGIC
# MAGIC For more information on how to read Shapley values, see the [SHAP documentation](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html).
# MAGIC
# MAGIC > **NOTE:** SHAP run may take a long time with the datetime columns in the dataset.

# COMMAND ----------

# Set this flag to True and re-run the notebook to see the SHAP plots
shap_enabled = False

# COMMAND ----------

if shap_enabled:
    mlflow.autolog(disable=True)
    mlflow.sklearn.autolog(disable=True)
    from shap import KernelExplainer, summary_plot
    # Sample background data for SHAP Explainer. Increase the sample size to reduce variance.
    train_sample = X_train.sample(n=min(100, X_train.shape[0]), random_state=651849752)

    # Sample some rows from the validation set to explain. Increase the sample size for more thorough results.
    example = X_val.sample(n=min(100, X_val.shape[0]), random_state=651849752)

    # Use Kernel SHAP to explain feature importance on the sampled rows from the validation set.
    predict = lambda x: model.predict(pd.DataFrame(x, columns=X_train.columns))
    explainer = KernelExplainer(predict, train_sample, link="identity")
    shap_values = explainer.shap_values(example, l1_reg=False, nsamples=500)
    summary_plot(shap_values, example)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference
# MAGIC [The MLflow Model Registry](https://docs.databricks.com/applications/mlflow/model-registry.html) is a collaborative hub where teams can share ML models, work together from experimentation to online testing and production, integrate with approval and governance workflows, and monitor ML deployments and their performance. The snippets below show how to add the model trained in this notebook to the model registry and to retrieve it later for inference.
# MAGIC
# MAGIC > **NOTE:** The `model_uri` for the model already trained in this notebook can be found in the cell below
# MAGIC
# MAGIC ### Register to Model Registry
# MAGIC ```
# MAGIC model_name = "Example"
# MAGIC
# MAGIC model_uri = f"runs:/{ mlflow_run.info.run_id }/model"
# MAGIC registered_model_version = mlflow.register_model(model_uri, model_name)
# MAGIC ```
# MAGIC
# MAGIC ### Load from Model Registry
# MAGIC ```
# MAGIC model_name = "Example"
# MAGIC model_version = registered_model_version.version
# MAGIC
# MAGIC model_uri=f"models:/{model_name}/{model_version}"
# MAGIC model = mlflow.pyfunc.load_model(model_uri=model_uri)
# MAGIC model.predict(input_X)
# MAGIC ```
# MAGIC
# MAGIC ### Load model without registering
# MAGIC ```
# MAGIC model_uri = f"runs:/{ mlflow_run.info.run_id }/model"
# MAGIC
# MAGIC model = mlflow.pyfunc.load_model(model_uri=model_uri)
# MAGIC model.predict(input_X)
# MAGIC ```

# COMMAND ----------

# model_uri for the generated model
print(f"runs:/{ mlflow_run.info.run_id }/model")