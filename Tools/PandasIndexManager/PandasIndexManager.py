import pandas as pd
import numpy as np


class PandasIndexManager:
    def __init__(self):
        pass

    @staticmethod
    def create_empty_exp_indexed_1d_df(name):
        df = pd.DataFrame(columns=['id_exp', name]).set_index(['id_exp'])
        return df.sort_index()

    @staticmethod
    def create_empty_exp_indexed_2d_df(xname, yname):
        df = pd.DataFrame(columns=['id_exp', xname, yname]).set_index(['id_exp'])
        return df.sort_index()

    @staticmethod
    def create_empty_exp_ant_indexed_df(name):
        df = pd.DataFrame(columns=['id_exp', 'id_ant', name]).set_index(['id_exp', 'id_ant'])
        return df.sort_index()

    @staticmethod
    def create_empty_exp_frame_indexed_1d_df(name):
        df = pd.DataFrame(columns=['id_exp', 'frame', name]).set_index(['id_exp', 'frame'])
        return df.sort_index()

    @staticmethod
    def create_empty_exp_frame_indexed_2d_df(xname, yname):
        df = pd.DataFrame(columns=['id_exp', 'frame', xname, yname]).set_index(['id_exp', 'frame'])
        return df.sort_index()

    @staticmethod
    def create_empty_exp_ant_frame_indexed_1d_df(name):
        df = pd.DataFrame(columns=['id_exp', 'id_ant', 'frame', name]).set_index(['id_exp', 'id_ant', 'frame'])
        return df.sort_index()

    @staticmethod
    def create_empty_exp_ant_frame_indexed_2d_df(xname, yname):
        df = pd.DataFrame(columns=['id_exp', 'id_ant', 'frame', xname, yname]).set_index(['id_exp', 'id_ant', 'frame'])
        return df.sort_index()

    @staticmethod
    def convert_to_exp_indexed_df(array, name):
        df = pd.DataFrame(array, columns=['id_exp', name])
        df.set_index(['id_exp'], inplace=True)
        return df.sort_index()

    @staticmethod
    def convert_to_exp_ant_indexed_df(array, name):
        df = pd.DataFrame(array, columns=['id_exp', 'id_ant', name])
        df.set_index(['id_exp', 'id_ant'], inplace=True)
        return df.sort_index()

    @staticmethod
    def convert_to_exp_frame_indexed_1d_df(array, name):
        df = pd.DataFrame(array, columns=['id_exp', 'frame', name])
        df.set_index(['id_exp', 'frame'], inplace=True)
        return df.sort_index()

    @staticmethod
    def convert_to_exp_frame_indexed_2d_df(array, xname, yname):
        df = pd.DataFrame(array, columns=['id_exp', 'frame', xname, yname])
        df.set_index(['id_exp', 'frame'], inplace=True)
        return df.sort_index()

    @staticmethod
    def convert_to_exp_ant_frame_indexed_1d_df(array, name):
        df = pd.DataFrame(array, columns=['id_exp', 'id_ant', 'frame', name])
        df.set_index(['id_exp', 'id_ant', 'frame'], inplace=True)
        return df.sort_index()

    @staticmethod
    def convert_to_exp_ant_frame_indexed_2d_df(array, xname, yname):
        df = pd.DataFrame(array, columns=['id_exp', 'id_ant', 'frame', xname, yname])
        df.set_index(['id_exp', 'id_ant', 'frame'], inplace=True)
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
    def get_array_id_exp(df_arg):
        if 'id_exp' not in df_arg.index.names:
            raise IndexError('df does not have id_exp as index')
        else:
            idx_set = set(df_arg.index.get_level_values('id_exp'))
            return np.array(sorted(idx_set), dtype=int)

    @staticmethod
    def get_array_id_exp_ant(df_arg):
        if 'id_exp' not in df_arg.index.names or 'id_ant' not in df_arg.index.names:
            raise IndexError('df does not have id_exp or id_ant as index')
        else:
            id_exps = df_arg.index.get_level_values('id_exp')
            id_ants = df_arg.index.get_level_values('id_ant')
            idx_set = set(list(zip(id_exps, id_ants)))
            return np.array(sorted(idx_set), dtype=int)

    @staticmethod
    def get_array_id_exp_frame(df_arg):
        if 'id_exp' not in df_arg.index.names or 'frame' not in df_arg.index.names:
            raise IndexError('df_arg does not have id_exp or frame as index')
        else:
            id_exps = df_arg.index.get_level_values('id_exp')
            frames = df_arg.index.get_level_values('frame')
            idx_set = set(list(zip(id_exps, frames)))
            return np.array(sorted(idx_set), dtype=int)

    @staticmethod
    def get_array_id_exp_ant_frame(df_arg):
        if 'id_exp' not in df_arg.index.names or 'id_ant' not in df_arg.index.names or 'frame' not in df_arg.index.names:
            raise IndexError('df does not have id_exp or id_ant or frame as index')
        else:
            idx_set = set(df_arg.index)
            return np.array(sorted(idx_set), dtype=int)

    def get_dict_id_exp_ant(self, df):
        index_array = self.get_array_id_exp_ant(df)
        res = dict()
        for (id_exp, id_ant) in index_array:
            if id_exp in res:
                res[id_exp].append(id_ant)
            else:
                res[id_exp] = [id_ant]
        for id_exp in res:
            res[id_exp].sort()
        return res

    def get_dict_id_exp_frame(self, df):
        index_array = self.get_array_id_exp_frame(df)
        res = dict()
        for (id_exp, frame) in index_array:
            if id_exp in res:
                res[id_exp].append(frame)
            else:
                res[id_exp] = [frame]
        for id_exp in res:
            res[id_exp].sort()
        return res

    @staticmethod
    def get_array_all_indexes(df_arg):
        idx_set = set(df_arg.index)
        return np.array(sorted(idx_set), dtype=int)

    def get_dict_id_exp_ant_frame(self, df):
        index_array = self.get_array_id_exp_ant_frame(df)
        res = dict()
        for (id_exp, id_ant, frame) in index_array:
            if id_exp in res:
                if id_ant in res[id_exp]:
                    res[id_exp][id_ant].append(frame)
                else:
                    res[id_exp][id_ant] = [frame]
            else:
                res[id_exp] = dict()
                res[id_exp][id_ant] = [frame]
        for id_exp in res:
            for id_ant in res[id_exp]:
                res[id_exp][id_ant].sort()
        return res

    def get_dict_id_exp_frame_ant(self, df):
        index_array = self.get_array_id_exp_ant_frame(df)
        res = dict()
        for (id_exp, id_ant, frame) in index_array:
            if id_exp in res:
                if frame in res[id_exp]:
                    res[id_exp][frame].append(id_ant)
                else:
                    res[id_exp][frame] = [id_ant]
            else:
                res[id_exp] = dict()
                res[id_exp][frame] = [id_ant]
        for id_exp in res:
            for frame in res[id_exp]:
                res[id_exp][frame].sort()
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
