import gdal
from Structure import Model
import numpy as np
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
import os

# encoding = utf-8

# @Author  ：Lecheng Wang
# @Time    : ${2025/7/15} ${19:02}
# @function: 用于计算每轮预测准确率时，前向网络权值装载和预测分割结果的生成


from Structure import Model

import os
import gdal
import numpy as np
gdal.UseExceptions()


# Read multi_spectral image raster data and geo_coordinate info.
def read_multiband_image(image_path):
    dataset = gdal.Open(image_path)
    if dataset is None:
        raise FileNotFoundError(f"Can't Open file in : {image_path}")

    bands  = dataset.RasterCount
    height = dataset.RasterYSize
    width  = dataset.RasterXSize
    image  = np.zeros((bands, height, width), dtype=np.float32)

    for i in range(bands):
        band         = dataset.GetRasterBand(i+1)
        image[i,:,:] = band.ReadAsArray()

    geotransform = dataset.GetGeoTransform()
    projection   = dataset.GetProjection()    
    dataset      = None
    return image, geotransform, projection


# Transfer tiff prediction result to shape file
def raster_tif_to_shp(tif_path, shp_dir, ignore_class=0, epsg_code=None):
    label_map = {
        1: "clean_glacier",
        2: "debris_glacier"
    }

    if not os.path.exists(shp_dir):
        os.makedirs(shp_dir)

    base_name = os.path.splitext(os.path.basename(tif_path))[0]
    shp_path  = os.path.join(shp_dir, f"{base_name}.shp")

    with rasterio.open(tif_path) as src:
        image      = src.read(1)
        transform  = src.transform
        raster_crs = src.crs

    unique_values = np.unique(image)
    unique_values = unique_values[unique_values != ignore_class]

    if len(unique_values) == 0:
        print("No valid classes found in the prediction.")
        return

    all_geometries = []
    all_class_ids  = []
    all_labels     = []

    for class_value in unique_values:
        mask = image == class_value
        shapes_generator = shapes(image.astype(np.uint8), mask=mask, transform=transform)
        for geom, value in shapes_generator:
            if value == class_value:
                all_geometries.append(shape(geom))
                all_class_ids.append(int(class_value))
                all_labels.append(label_map.get(int(class_value), "unknown"))

    if len(all_geometries) == 0:
        print("No geometries extracted.")
        return

    gdf = gpd.GeoDataFrame({
        "class_id": all_class_ids,
        "label": all_labels,
        "geometry": all_geometries
    }, crs=raster_crs)

    if epsg_code:
        gdf = gdf.to_crs(epsg=f"EPSG:{epsg_code}")

    gdf.to_file(shp_path, driver="ESRI Shapefile")
    print(f"Shapefile saved in: {shp_path}, total objects: {len(gdf)}")

# Save prediction as tif file.
def save_prediction_as_geotiff(prediction, geotransform, projection, output_path):
    height, width = prediction.shape
    driver        = gdal.GetDriverByName('GTiff')
    dataset       = driver.Create(output_path, width, height, 1, gdal.GDT_Byte)
    
    dataset.SetGeoTransform(geotransform)
    dataset.SetProjection(projection)
    dataset.GetRasterBand(1).WriteArray(prediction)
    dataset.FlushCache()
    dataset = None
    print(f"Prediction already saved with GeoTIFF in : {output_path}")


def main(model_cfg, model_path, input_image_path, output_tiff_path, output_png_path, shape_out_dir):
    model = Model(model_path=model_path, bands=model_cfg["bands"], num_class=model_cfg["num_classes"], 
                  model_type=model_cfg["model_type"], backbone=model_cfg["backbone_type"], atten_type=model_cfg["atten_type"])

    try:
        image, geotransform, projection = read_multiband_image(input_image_path)
        print(f"Image dimension: {image.shape[1]}x{image.shape[2]}, bands num: {image.shape[0]}")
    except Exception as e:
        print(f"Faile to read image: {e}")
        return

    print("Prediction on going....")
    try:
        predicted_result     = model.predict_large_image(image[:7,:,:])
        predicted_png_result = model.get_large_predict_png(image[:7,:,:])
    except Exception as e:
        print(f"Faile to predict image: {e}")
        return

    try:
        predicted_png_result.save(output_png_path)
        save_prediction_as_geotiff(predicted_result, geotransform, projection, output_tiff_path)
    except Exception as e:
        print(f"Faile to save prediction: {e}")
        return
    
    try:
        raster_tif_to_shp(tif_path=output_tiff_path, shp_dir=shape_out_dir)
    except Exception as e:
        print(f"Failed to transfer into shapefile: {e}")
        return
    
    

if __name__ == "__main__":
    result_dir    = os.path.join("./output/")
    shp_dir       = os.path.join("./output/shape/")
    os.makedirs(result_dir, exist_ok=True)

    model_path    = "./pth_files/7bands_upernet-epoch100-loss0.239-val_loss0.243.pth"
    img_in_path   = "./test_sample/Test2_sample_10bands.tif"

    cfg = {
        "bands" : 7,
        "num_classes" : 3,
        "model_type" : 'upernet',
        "backbone_type": 'mobilenet',
        "atten_type": None
    }

    img_name      = os.path.splitext(os.path.basename(img_in_path))[0]
    seg_out_path  = os.path.join(result_dir, f"seg_{img_name}_class.tif")
    seg_png_path  = os.path.join(result_dir, f"seg_{img_name}_png.tiff")
    main(cfg, model_path, img_in_path, seg_out_path, seg_png_path, shp_dir)