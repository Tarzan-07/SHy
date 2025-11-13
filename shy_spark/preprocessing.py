from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_list, array_distinct, substring
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.functions import ArrayType, IntegerType
import logging

logging.basicConfig(level=logging.DEBUG)

icd_code_col = 'icd9_code'
patient_col = 'patient_id'
visit_col = 'visit_id'
code_trim_len = 3

spark = SparkSession.builder.appName('hypergraph_processing_pipeline').getOrCreate()

try:
    admit = spark.read.csv('./mimic-iii-clinical-database-demo-1.4/ADMISSIONS.csv')
    diag = spark.read.csv('./mimic-iii-clinical-database-demo-1.4/DIAGNOSES_ICD.csv')
    patients = spark.read.csv('./mimic-iii-clinical-database-demo-1.4/PATIENTS.csv')

    try: 
        df_adm_pat = admit.join(patients.select('PATIENT_ID'), on='PATIENT_ID',how='inner')
        df = df_adm_pat.join(diag.select('PATIENT_ID', 'VISIT_ID'), ON='VISIT_ID', how='inner') 
    except:
        logging.exception("An error occured in combining")
except:
    logging.exception("An error has occured in read csv")

critical_columns = ['PATIENT_ID', 'VISIT_ID', 'ICD_CODE']

df_final = df.dropna(subset=critical_columns)

df_codes_generalized = df_final.withColumn('general_code', substring(col(icd_code_col),  1, code_trim_len))

df_visit_nodes = df_codes_generalized.groupBy(patient_col, visit_col).agg(
    array_distinct(collect_list(col('generalized_code'))).alias('visit_codes')
)

all_codes = df_codes_generalized.select('generalized_code').distinct().collect()

unique_codes_list = [row[0] for row in all_codes]

code_to_idx =  {code: i for i, code in enumerate(unique_codes_list)}

num_features = len(unique_codes_list)

broadcast_map = spark.sparkContext.broadcast(code_to_idx)

def code_to_indices_func(codes):
    if codes is None:
        return []
    return [broadcast_map.value[code] for code in codes if codes in broadcast_map.value]

udf_code_to_idx = spark.udf.register(
    'codes_to_idx',  code_to_indices_func, ArrayType(IntegerType())
)

df_nodes_feat = df_visit_nodes.withColumn(
    'FEATURE_IDX', udf_code_to_idx(col('VISIT_CODES'))
)

df_final_nodes = df_nodes_feat.withColumn(
    'NODE_ID', monotonically_increasing_id()
).select(patient_col, visit_col, 'NODE_ID',  'FEATURE_IDX').orderBy('NODE_ID')

df_hyper = df_final_nodes.groupBy(patient_col).agg(collect_list(col('NODE_ID')).alias('NODE_IDX_IN_HYPEREDGE')
).select('NODE_IDX_IN_HYPEREDGEDE')

node_feature_idx_pyg = df_final_nodes.select('FEATURE_IDX').collect()

hyperedge_list_pyg =  df_hyper.collect()

spark.stop()