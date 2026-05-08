import unittest

from tsdatasets import TimeSeriesDataModule, TimeSeriesDataset, TimeSeriesRecord


class TestDatasetPrimitives(unittest.TestCase):
    def test_dataset_sequence(self) -> None:
        records = [
            TimeSeriesRecord(values=(1.0, 2.0), target=1.0),
            TimeSeriesRecord(values=(3.0, 4.0), target=0.0),
        ]
        dataset = TimeSeriesDataset(records, name="sample")

        self.assertEqual(dataset.name, "sample")
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[1].values, (3.0, 4.0))

    def test_record_validates_lengths(self) -> None:
        with self.assertRaises(ValueError):
            TimeSeriesRecord(values=(1.0, 2.0), timestamps=("2020-01-01",))

    def test_datamodule_holds_splits(self) -> None:
        train = TimeSeriesDataset([TimeSeriesRecord(values=(1.0,))], name="train")
        valid = TimeSeriesDataset([TimeSeriesRecord(values=(2.0,))], name="valid")

        datamodule = TimeSeriesDataModule(train=train, validation=valid)

        self.assertIs(datamodule.train, train)
        self.assertIs(datamodule.validation, valid)
        self.assertIsNone(datamodule.test)


if __name__ == "__main__":
    unittest.main()
