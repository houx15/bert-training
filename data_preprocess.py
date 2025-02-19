"""
author: Hou Yuxin
date: 2022-06-16
"""

import os
import pandas as pd
import json

from sklearn.model_selection import train_test_split


class DataProcess(object):
    """
    Base class for data process.
    """

    def __init__(
        self,
        dataset_dir: str,
        dataset_file: str,
        task_type: str,
        src_type: str = "weibo",
        force_update: bool = False,
        split_ratio: list = [0.6, 0.2, 0.2],
    ) -> None:
        """
        dataset_dir: str, path of dataset directory
        dataset_file: str, the filename of a datafile
        task_type: str, regression or binary
        force_update: bool, when it equals to False, program would use train/validate/test-{task_type}.csv as default datafile
        split_ratio: list, we split the dataset to train, validate, test dataset according to the ratio. Each item should be float and the sum of them should equal to 1
        """
        # task_map = {
        #     "ternary": "binary",
        #     "regression": "regression",
        #     "binary": "binary"
        # }
        import os

        self.dataset_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), dataset_dir
        )
        self.dataset_file = dataset_file
        self.task_type = task_type  # task_map[task_type]
        self.src_type = src_type
        self.force_update = force_update
        self.split_ratio = split_ratio

    def split_df_by_ratio(self, df):
        if sum(self.split_ratio) != 1:
            raise ValueError("Error: the sum of split ratio is larger than 1")
        if len(self.split_ratio) < 3:
            raise ValueError(
                "Error: split_ratio doesn't provide the ratio of train, validate and test dataset"
            )
        print("type", df, type(df))

        train_df = df.sample(frac=self.split_ratio[0])
        remained_df = df[~df.index.isin(train_df.index)]

        validate_df = remained_df.sample(
            frac=self.split_ratio[1] / (self.split_ratio[1] + self.split_ratio[2])
        )
        test_df = remained_df[~remained_df.index.isin(validate_df.index)]

        return train_df, validate_df, test_df

    def split_df_by_label(self, df):
        """
        Code from https://stackoverflow.com/questions/50781562/stratified-splitting-of-pandas-dataframe-into-training-validation-and-test-set
        Splits a Pandas dataframe into three subsets (train, val, and test)
        following fractional ratios provided by the user, where each subset is
        stratified by the values in a specific column (that is, each subset has
        the same relative frequency of the values in the column). It performs this
        splitting by running train_test_split() twice.

        Parameters
        ----------
        df_input : Pandas dataframe
            Input dataframe to be split.
        stratify_colname : str
            The name of the column that will be used for stratification. Usually
            this column would be for the label.
        frac_train : float
        frac_val   : float
        frac_test  : float
            The ratios with which the dataframe will be split into train, val, and
            test data. The values should be expressed as float fractions and should
            sum to 1.0.
        random_state : int, None, or RandomStateInstance
            Value to be passed to train_test_split().

        Returns
        -------
        df_train, df_val, df_test :
            Dataframes containing the three splits.
        """

        if sum(self.split_ratio) != 1:
            raise ValueError("Error: the sum of split ratio is larger than 1")
        if len(self.split_ratio) < 3:
            raise ValueError(
                "Error: split_ratio doesn't provide the ratio of train, validate and test dataset"
            )
        print("type", df, type(df))

        X = df  # Contains all columns.
        y = df[["label"]]  # Dataframe of just the column on which to stratify.

        frac_train, frac_val, frac_test = self.split_ratio

        # Split original dataframe into train and temp dataframes.
        df_train, df_temp, y_train, y_temp = train_test_split(
            X, y, stratify=y, test_size=(1.0 - frac_train), random_state=None
        )

        # Split the temp dataframe into val and test dataframes.
        relative_frac_test = frac_test / (frac_val + frac_test)

        if relative_frac_test > 0:
            df_val, df_test, y_val, y_test = train_test_split(
                df_temp,
                y_temp,
                stratify=y_temp,
                test_size=relative_frac_test,
                random_state=None,
            )
        else:
            df_val = df_temp
            df_test = pd.DataFrame(columns=df_val.columns)

        assert len(df) == len(df_train) + len(df_val) + len(df_test)

        return df_train, df_val, df_test

    def save_df(self, train_df, validate_df, test_df, total_df):
        train_df.to_csv(f"{self.dataset_dir}train-{self.task_type}.csv", index=False)
        validate_df.to_csv(
            f"{self.dataset_dir}validate-{self.task_type}.csv", index=False
        )
        test_df.to_csv(f"{self.dataset_dir}test-{self.task_type}.csv", index=False)
        total_df.to_csv(f"{self.dataset_dir}total-{self.task_type}.csv", index=False)
        print("saving finished")
        return True

    def get_cached_df(self):
        try:
            train_df = pd.read_csv(f"{self.dataset_dir}train-{self.task_type}.csv")
            validate_df = pd.read_csv(
                f"{self.dataset_dir}validate-{self.task_type}.csv"
            )
            test_df = pd.read_csv(f"{self.dataset_dir}test-{self.task_type}.csv")
            total_df = pd.read_csv(f"{self.dataset_dir}total-{self.task_type}.csv")
            return train_df, validate_df, test_df, total_df
        except:
            return None, None, None, None

    def get_merged_dataset(self, split=False):
        df_list = []
        for single_file in self.dataset_files:
            single_df = self.single_file_handler(single_file)
            df_list.append(single_df)
        dataset_df = pd.concat(df_list)
        if split:
            train_df, validate_df, test_df = self.split_df_by_label(dataset_df)
            return train_df, validate_df, test_df, dataset_df
        return dataset_df
    
    def dataset_handler(self, dataset_file):
        if not dataset_file.endswith(".parquet"):
            raise ValueError("The dataset file should be a parquet file.")
        dataset_path = os.path.join(self.dataset_dir, dataset_file)
        dataset_df = pd.read_parquet(dataset_path)
        dataset_df = dataset_df[["weibo_id", "text", "opinion"]]
        namemapping = {
            "weibo_id": "id",
            "text": "text",
            "opinion": "label",
        }
        dataset_df.rename(columns=namemapping, inplace=True)
        if self.task_type == "binary":
            dataset_df["label"] = dataset_df["label"].apply(
                lambda x: 1 if x is not None else 0
            )
        else:
            dataset_df = dataset_df.dropna(subset=["label"])
        return dataset_df

    def get_dataset(self):
        """
        Return a 3-item tuple
        (train_dataset, validate_dataset, test_dataset)
        Each element is a pandas DataFrame
        """
        train_df, validate_df, test_df, dataset_df = (
            None,
            None,
            None,
            None,
        )
        if self.force_update is False:
            train_df, validate_df, test_df, dataset_df = (
                self.get_cached_df()
            )

        if train_df is None:
            dataset_df = self.dataset_handler(self.dataset_file)
            train_df, validate_df, test_df = self.split_df_by_label(dataset_df)
            dataset_df = pd.concat([train_df, validate_df, test_df])
        self.save_df(train_df, validate_df, test_df, dataset_df)

        return train_df, validate_df, test_df, dataset_df
