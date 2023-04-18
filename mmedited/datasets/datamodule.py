from typing import Any, Optional, Union, Dict, List

from pytorch_lightning import LightningDataModule as PLDataModule
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate


class DataModule(PLDataModule):
    def __init__(
        self,
        train_dataset=None,
        val_dataset=None,
        test_dataset=None,
        train_dataloader_cfg: Dict[str, Any]=None,
        val_dataloader_cfg: Dict[str, Any]=None,
        test_dataloader_cfg: Dict[str, Any]=None,
    ):
        """
        :author: Luca Actis Grosso
        :param train_dataset: Dataset that will be used during the training step
        :param val_dataset: Dataset that will be used during the validation step
        :param test_dataset: Dataset that will be used during the test step
        :param train_dataloader_cfg: dictionary containing the parameters of the train dataloader
        :param val_dataloader_cfg: dictionary containing the parameters of the val dataloader
        :param test_dataloader_cfg: dictionary containing the parameters of the test dataloader
        """
        super().__init__()

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.train_dataloader_cfg = train_dataloader_cfg if train_dataloader_cfg else {}
        self.val_dataloader_cfg = val_dataloader_cfg if val_dataloader_cfg else {}
        self.test_dataloader_cfg = test_dataloader_cfg if test_dataloader_cfg else {}
        self.save_hyperparameters()

    @staticmethod
    def collate_fn_supports_none(batch: Any) -> Any:
        """
        If there is an issue with a particular sample, the dataset returns None, and we skip it from the batch.
        :author: Sergiu Neamtu
        """
        batch = list(filter(lambda x: x is not None, batch))
        #for el in batch:
        #    del el["meta"]
        return default_collate([b["lq"] for b in batch]), default_collate([b["gt"] for b in batch])

    def train_dataloader(self) -> DataLoader:
        """
        :author: Sergiu Neamtu
        :return: The :class:`torch.utils.data.DataLoader` used for training.
        """
        if self.train_dataset is None:
            super().train_dataloader()
        return DataLoader(
            dataset=self.train_dataset,  # type: ignore
            **self.train_dataloader_cfg,
            collate_fn=self.collate_fn_supports_none
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        """
        :author: Sergiu Neamtu
        :return: The :class:`torch.utils.data.DataLoader` used for validation.
        """
        if self.val_dataset is None:
            super().val_dataloader()
        if isinstance(self.val_dataset, list):
            dataloaders = []
            for val_dataset in self.val_dataset:
                dataloaders.append(
                    DataLoader(
                        dataset=val_dataset,  # type: ignore
                        **self.val_dataloader_cfg,
                        collate_fn=self.collate_fn_supports_none
                    )
                )
            return dataloaders
        else:
            return DataLoader(
                dataset=self.val_dataset,  # type: ignore
                **self.val_dataloader_cfg,
                collate_fn=self.collate_fn_supports_none
            )

    def test_dataloader(self) -> DataLoader:
        """
        :author: Sergiu Neamtu
        :return: The :class:`torch.utils.data.DataLoader` used for testing.
        """
        if self.test_dataset is None:
            super().test_dataloader()
        return DataLoader(
            dataset=self.test_dataset,  # type: ignore
            **self.test_dataloader_cfg,
            collate_fn=self.collate_fn_supports_none
        )
