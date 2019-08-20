from omrdatasettools.downloaders.MuscimaPlusPlusDatasetDownloader import MuscimaPlusPlusDatasetDownloader
from omrdatasettools.image_generators.MuscimaPlusPlusMaskImageGenerator import \
    MuscimaPlusPlusMaskImageGenerator, MaskType

if __name__ == '__main__':
    downloader = MuscimaPlusPlusDatasetDownloader(dataset_version="2.0")
    downloader.download_and_extract_dataset("muscima_pp")

    mask_generator = MuscimaPlusPlusMaskImageGenerator()
    mask_generator.render_node_masks("muscima_pp", "muscima_pp_masks", MaskType.STAFF_BLOBS_INSTANCE_SEGMENTATION)
