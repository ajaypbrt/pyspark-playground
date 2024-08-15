from brtdevkit.data import Dataset
from pyspark.sql import SparkSession

from pyspark.ml import Transformer
from pyspark.ml.util import Identifiable

from pyspark.ml import Transformer
from pyspark.ml.util import Identifiable
from pyspark.sql.functions import udf
from pyspark.sql.types import StructType, StructField, ArrayType, FloatType, IntegerType
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import StringType
from PIL import Image
import boto3
import io
import numpy as np
import base64
import json
from brtdl.data.labelmap import add_ground_label, generate_correspondence_map, validate_remap_compatibility
from brtdl.data._sync_labelmap import labelmap_remap
from brtdl.data import SampleKeys
import imgaug.augmenters as iaa
from brtdl.transforms import custom_transforms
from brtdl.data.labelmap import retrieve_labelmap
from pyspark.sql.functions import col
from brtdl.transforms import SegmentationTransform


# Custom pyspark type containing numpy bytes and the dtype + dimensions of the image
PySparkNumpyType = StructType([
    StructField("image_data", StringType(), nullable=False),  # Base64 encoded bytes
    StructField("dtype", StringType(), nullable=False),  # numpy dtype
    StructField("dimensions", ArrayType(IntegerType()), nullable=False)  # dimensions tuple
])

AnnotationType = StructType([
    StructField("annotation_data", PySparkNumpyType, nullable=False),
    StructField("labelmap", StringType(), nullable=False)
])

# Convert a numpy array to PySparkNumpyType
def numpy_to_pyspark(ndarray):
    return (base64.b64encode(ndarray.tobytes()).decode('utf-8'),
        ndarray.dtype.name,
        tuple(ndarray.shape),)

def pyspark_to_numpy(pyspark_ndarray):
    return np.frombuffer(base64.b64decode(pyspark_ndarray.image_data), pyspark_ndarray.dtype).reshape(pyspark_ndarray.dimensions)


# Input S3 location => Output base64 encoded bytes of the image
class S3ImageDownloader(Transformer, HasOutputCol):
    def __init__(self, s3_key_col, s3_bucket_col, output_col):
        super().__init__()
        self.s3_key_col = s3_key_col
        self.s3_bucket_col = s3_bucket_col
        self.outputCol = output_col

    def _transform(self, dataset):
        # Define UDF to process each row
        def im_download(s3_key, s3_bucket):
            s3_client = boto3.client('s3')
            # Download image from S3
            response = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
            img_data = response['Body'].read()
            img = np.array(Image.open(io.BytesIO(img_data)))
            return numpy_to_pyspark(img)

        im_download_udf = udf(lambda s3_key, s3_bucket: im_download(s3_key, s3_bucket), PySparkNumpyType)
        return dataset.withColumn(self.outputCol, im_download_udf(dataset[self.s3_key_col], dataset[self.s3_bucket_col]))


class LabelmapSync(Transformer, Identifiable):
    def __init__(self, to_labelmap, correspondence_map, fix_ground):
        super().__init__()
        self.to_labelmap = retrieve_labelmap(to_labelmap)
        self.correspondence_map = retrieve_labelmap(correspondence_map)
        self.fix_ground = fix_ground

    def _transform(self, dataset):
        def sync_labelmap(annotation_data, labelmap):
            from_labelmap = {int(key): val for key, val in labelmap.items()}

            if self.fix_ground:
                add_ground_label(from_labelmap)

            # Trivial case of labelmaps being equal.
            if from_labelmap == self.to_labelmap:
                return (annotation_data, self.to_labelmap)

            # Check compatibility of current labelmap
            if not validate_remap_compatibility(from_labelmap=from_labelmap,
                                                to_labelmap=self.to_labelmap,
                                                correspondence_map=self.correspondence_map):
                raise ValueError("Invalid labelmap remapping.\n"
                                f"\tFROM labels : {from_labelmap.values()}\n"
                                f"\tTO labels : {self.to_labelmap.values()}\n"
                                f"\tFROM:TO remapping : {self.correspondence_map}"
                                )

            # Generate index map for current labelmap if compatible
            c_idx_map = generate_correspondence_map(from_labelmap, self.to_labelmap, self.correspondence_map)

            annotation = pyspark_to_numpy(annotation_data)
            synced_annotation = labelmap_remap(annotation, c_idx_map)
            return (numpy_to_pyspark(synced_annotation), self.to_labelmap)
    
        sync_labelmap_udf = udf(lambda annotation_data, labelmap: sync_labelmap(annotation_data, labelmap), AnnotationType)
        return dataset.withColumn(
            "synced_annotation_data",
            sync_labelmap_udf(dataset["annotation_data"], dataset["annotation_pixelwise_0_label_map"])
        ).drop("annotation_data").select(
            "*",
            col("synced_annotation_data.annotation_data").alias("annotation_data"), 
            col("synced_annotation_data.labelmap").alias("annotation_pixelwise_0_label_map")
        ).drop("synced_annotation_data")

_image_data_type_transform_map = {
    'uint8': custom_transforms.ToUint8,
    'float16': custom_transforms.ToFloat16,
    'float32': custom_transforms.ToFloat32
}

class NpzPackager(Transformer, HasOutputCol):
    def __init__(self, input_image_modes, output_image_modes, input_shape, image_data_type):
        super().__init__()
        self.outputCol = "npz_data"
        self.input_image_modes = input_image_modes
        self.output_image_modes = output_image_modes
        self.input_shape = input_shape
        self.image_data_type = image_data_type
        self.my_transform = SegmentationTransform(iaa.Sequential([
            custom_transforms.ImageFormat(input_mode=self.input_image_modes, output_mode=self.output_image_modes),
            custom_transforms.ToFloat32(),
            custom_transforms.ResizeImageLabel(image_size=self.input_shape, label_size=self.input_shape),
            _image_data_type_transform_map[self.image_data_type](),
        ]))

    def _transform(self, dataset):
        def pack_npz(image_data, annotation_data):
            image = pyspark_to_numpy(image_data)
            annotation = pyspark_to_numpy(annotation_data)
            sample = {
                SampleKeys.image: image,
                SampleKeys.label: annotation
            }
            sample = self.my_transform(sample)
            bytes_io = io.BytesIO()
            np.savez_compressed(bytes_io, **sample)
            return base64.b64encode(bytes_io.getvalue()).decode('utf-8')
        pack_npz_udf = udf(lambda annotation_data, annotation_labelmap: pack_npz(annotation_data, annotation_labelmap), StringType())
        return dataset.withColumn(self.outputCol, pack_npz_udf(dataset["image_data"], dataset["annotation_data"]))


class ImageDimensionExtractor(Transformer, HasInputCol, HasOutputCol):
    def __init__(self, inputCol, outputCol):
        super(ImageDimensionExtractor, self).__init__()
        self._setDefault(inputCol=inputCol, outputCol=outputCol)
    
    def _transform(self, dataset):
        input_col = self.getInputCol()
        output_col = self.getOutputCol()

        # Define a UDF to extract the dimension of a vector
        def get_dimensions(image_data):
            img = pyspark_to_numpy(image_data)
            return str(img.size)

        dimension_udf = udf(get_dimensions, StringType())

        # Apply the UDF to the input column and create a new column with the dimensions
        return dataset.withColumn(output_col, dimension_udf(dataset[input_col]))


ds = Dataset.retrieve('66b0f40dab0fb042364a859e')
dset_df = ds.to_dataframe()[['artifact_nrg_0_web_s3_key', 'artifact_nrg_0_web_s3_bucket', 'annotation_pixelwise_0_s3_key', 'annotation_pixelwise_0_s3_bucket', 'annotation_pixelwise_0_label_map']]
print(len(dset_df))
dset_df = dset_df.head(200)

spark = SparkSession.builder.appName(
    "Evaluation: Datasets"
).getOrCreate()

spark_df = spark.createDataFrame(dset_df).persist(storageLevel="DISK_ONLY")

image_download = S3ImageDownloader(s3_key_col='artifact_nrg_0_web_s3_key', s3_bucket_col='artifact_nrg_0_web_s3_bucket', output_col='image_data')
spark_df = image_download.transform(spark_df).persist(storageLevel="DISK_ONLY")

anno_download = S3ImageDownloader(s3_key_col='annotation_pixelwise_0_s3_key', s3_bucket_col='annotation_pixelwise_0_s3_bucket', output_col='annotation_data')
spark_df = anno_download.transform(spark_df).persist(storageLevel="DISK_ONLY")

labelmap_sync = LabelmapSync(to_labelmap="DEFAULT_LABELMAP", correspondence_map="DEFAULT_C_MAP", fix_ground=True)
spark_df = labelmap_sync.transform(spark_df).persist(storageLevel="DISK_ONLY")

npz_packager = NpzPackager(input_image_modes='nrg', output_image_modes='nrg', input_shape=(540, 960), image_data_type='float32')
spark_df = npz_packager.transform(spark_df).persist(storageLevel="DISK_ONLY")

# dimension_extractor = ImageDimensionExtractor(inputCol='nrg', outputCol='nrg_dimension')
# spark_df = dimension_extractor.transform(spark_df)

spark_df.select('npz_data').show()
