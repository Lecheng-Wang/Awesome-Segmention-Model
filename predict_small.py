# encoding = utf-8

# @Author  ：Lecheng Wang
# @Time    : 2025/7/15  19:02
# @function: 


from Structure import Model

import time
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


def main(model_cfg, model_path, input_image_path, output_tiff_path, output_png_path):
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
        start_time           = time.time()
        predicted_result     = model.predict_small_patch(image[:10,:,:])
        end_time             = time.time()
        inference_time_ms    = (end_time - start_time) * 1000  # transfer into ms
        print(f"Inference Time: {inference_time_ms:.2f} ms")

        predicted_png_result = model.get_small_predict_png(image[:10,:,:])
    except Exception as e:
        print(f"Faile to predict image: {e}")
        return

    try:
        predicted_png_result.save(output_png_path)
        save_prediction_as_geotiff(predicted_result, geotransform, projection, output_tiff_path)
    except Exception as e:
        print(f"Faile to save prediction: {e}")
    

if __name__ == "__main__":
    result_dir    = os.path.join("./output/")
    os.makedirs(result_dir, exist_ok=True)

    model_path    = "./pth_files/unet-epoch1-loss0.344-val_loss0.228.pth"
    img_in_path   = "./test_sample/365.tif"

    cfg = {
        "bands" : 10,
        "num_classes" : 3,
        "model_type" : 'unet',
        "backbone_type": 'resnet50',
        "atten_type": None
    }

    img_name      = os.path.splitext(os.path.basename(img_in_path))[0]
    seg_out_path  = os.path.join(result_dir, f"seg_{img_name}_class.tif")
    seg_png_path  = os.path.join(result_dir, f"seg_{img_name}_png.tiff")
    main(cfg, model_path, img_in_path, seg_out_path, seg_png_path)

