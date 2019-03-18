import pandas as pd
import numpy as np

from Tools.MiscellaneousTools.ArrayManipulation import turn_to_list


class PandasIndexManager:
    def __init__(self):
        pass

    @staticmethod
    def create_empty_df(index_names, column_names):
        index_names = turn_to_list(index_names)
        column_names = turn_to_list(column_names)
        df = pd.DataFrame(columns=index_names+column_names).set_index(index_names)
        return df

    @staticmethod
    def convert_array_to_df(array, index_names, column_names):
        index_names = turn_to_list(index_names)
        column_names = turn_to_list(column_names)
        df = pd.DataFrame(array, columns=index_names+column_names)
        df.set_index(index_names, inplace=True)
        return df.sort_index()

    @staticmethod
    def convert_df_to_array(df):
        return np.array(df.reset_index())

    @staticmethod
    def convert_pandas_series_to_df(df, name):
        df = pd.DataFrame(df)
        df.columns = [name]
        return df

    @staticmethod
    def get_row_of_idx_array(idx_array, df):
        return df.loc[list(map(tuple, np.array(idx_array))), :]

    @staticmethod
    def get_index_location(df, index_name):
        return np.where(np.array(list(df.index.names)) == index_name)[0][0]

    @staticmethod
    def get_unique_index_array(df, index_names=None):
        if index_names is None:
            idx_set = set(df.index)
            return np.array(sorted(idx_set))
        else:
            index_names = turn_to_list(index_names)
            list_names = list(df.index.names)
            for index_name in index_names:
                list_names.remove(index_name)

            idxs = df.index
            for names in list_names:
                idxs = idxs.droplevel(names)
            idx_set = set(idxs)
            return np.array(sorted(idx_set))

    @staticmethod
    def get_index_array(df, index_names=None):
        if index_names is None:
            return np.array(df.index)
        else:
            index_names = turn_to_list(index_names)
            idxs = []
            for index_name in list(df.index.names):
                if index_name in index_names:
                    idxs.append(list(df.index.get_level_values(index_name)))
            return np.array(idxs).T

    def get_index_dict(self, df, index_names):
        if len(index_names) not in [2, 3]:
            raise IndexError('Only one index name or to many index names')
        else:
            index_array = self.get_unique_index_array(df, index_names)

            res = dict()
            if len(index_names) == 2:

                for (idx1, idx2) in index_array:
                    if idx1 in res:
                        res[idx1].append(idx2)
                    else:
                        res[idx1] = [idx2]
                for idx1 in res:
                    res[idx1].sort()
            else:
                for (idx1, idx2, idx3) in index_array:
                    if idx1 in res:
                        if idx2 in res[idx1]:
                            res[idx1][idx2].append(idx3)
                        else:
                            res[idx1][idx2] = [idx3]
                    else:
                        res[idx1] = dict()
                        res[idx1][idx2] = [idx3]
                for idx1 in res:
                    for idx2 in res[idx1]:
                        res[idx1][idx2].sort()
            return res

    def get_dict_index_with_one_index_fixed(self, df, fixed_index_name, fixed_index_value):
        index_names = df.index.names
        loc_list = list(range(len(index_names)))
        idx_loc = self.get_index_location(df, fixed_index_name)
        loc_list.remove(idx_loc)

        index_array = self.get_index_array(df=df, index_names=index_names)
        index_array = index_array[index_array[:, idx_loc] == fixed_index_value, :]
        index_array = index_array[:, loc_list]

        res = dict()
        for (idx1, idx2) in index_array:
            if idx1 in res:
                res[idx1].append(idx2)
            else:
                res[idx1] = [idx2]

        for idx1 in res:
            res[idx1].sort()

        return res

    @staticmethod
    def concat_dfs(df1, df2):
        dfs_to_concat = [df1, df2]

        df_concat = pd.concat(dfs_to_concat)

        PandasIndexManager.index_as_type_int(df_concat)

        return df_concat

    @staticmethod
    def index_as_type_int(df):
        index_names = df.index.names
        df.reset_index(inplace=True)
        df[index_names] = df[index_names].astype(int)
        df.set_index(index_names, inplace=True)

    @staticmethod
    def add_index_level(df, index_level_name, index_values):
        index_names = df.index.names
        df.reset_index(inplace=True)
        df[index_level_name] = index_values
        df.set_index(index_names + [index_level_name], inplace=True)

    @staticmethod
    def remove_index_level(df, index_level_name):
        index_name_list = list(df.index.names)
        index_name_list.remove(index_level_name)
        df.reset_index(inplace=True)
        df.drop(index_level_name, axis=1, inplace=True)
        df.set_index(index_name_list, inplace=True)

    @staticmethod
    def rename_index_level(df, old_name, new_name):
        index_name_list = list(df.index.names)
        index_name_list[index_name_list.index(old_name)] = new_name
        df.index.names = index_name_list
