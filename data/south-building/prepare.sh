wget https://demuc.de/colmap/datasets/south-building.zip
unzip south-building.zip
rm south-building.zip

colmap image_undistorter \
--image_path south-building/images \
--input_path south-building/sparse \
--output_path . \
--output_type COLMAP \
--max_image_size 1500

rm -rf south-building

cd sparse
mkdir 0
cp *.bin 0