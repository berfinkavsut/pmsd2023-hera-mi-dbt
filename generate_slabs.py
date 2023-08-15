from program.slab_generation import SlabGeneration
from program.arg_parser import *


args = parse_arguments()
settings_config = load_settings(args)
slab_gen_config = load_slab_configs(args)

slab_gen = SlabGeneration(settings_config=settings_config)
slab_gen.init()

# Read DICOM filepaths from the input folder and generate the configurations
dicom_filepaths, dicom_num = slab_gen.list_dicom_filepaths()
slab_config_hash_codes, slab_config_num = slab_gen.generate_configurations(slab_configs=slab_gen_config,
                                                                           save_mode=True)

# Generate slabs for each DBT volume
for i, dicom_filepath in enumerate(dicom_filepaths):
    image, series_instance_uid = slab_gen.read_dicom(dicom_filepath=dicom_filepath,
                                                     current_number=(i+1),
                                                     total_number=dicom_num)

    # Generate slabs for each configuration
    for j, slab_config_hash_code in enumerate(slab_config_hash_codes):

        slab_config = slab_gen.read_config_json(config_hash_code=slab_config_hash_code)

        slabs = slab_gen.generate_slabs(image=image, config=slab_config)

        slab_gen.save_slabs(slabs=slabs,
                            config_hash_code=slab_config_hash_code,
                            series_instance_uid=series_instance_uid,
                            )
