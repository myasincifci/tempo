from fiftyone import Dataset
from fiftyone.types import COCODetectionDataset

def main():
    train_dataset = Dataset.from_dir(
        dataset_type=COCODetectionDataset,
        data_path="./datasets/ASL_mask/images",
        labels_path="./datasets/ASL_mask/annotations/instances_Train.json",
        include_id=True,
    )

    # test_dataset = fo.Dataset.from_dir(
    #     dataset_type=fo.types.COCODetectionDataset,
    #     data_path="./datasets/ASL_mask/images",
    #     labels_path="./datasets/ASL_mask/annotations/instances_Test.json",
    #     include_id=True,
    # )

    # session = fo.launch_app(test_dataset)

    a=1

if __name__ == "__main__":
    main()