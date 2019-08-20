from omrdatasettools.downloaders.MuscimaPlusPlusDatasetDownloader import MuscimaPlusPlusDatasetDownloader

if __name__ == '__main__':
    downloader = MuscimaPlusPlusDatasetDownloader(dataset_version="2.0")
    downloader.download_and_extract_dataset("muscima-pp")