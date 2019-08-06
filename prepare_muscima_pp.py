from omrdatasettools.downloaders.MuscimaPlusPlusDatasetDownloader import MuscimaPlusPlusDatasetDownloader

if __name__ == '__main__':
    downloader = MuscimaPlusPlusDatasetDownloader(version=2)
    downloader.download_and_extract_dataset("muscima-pp")