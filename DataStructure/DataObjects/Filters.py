from DataStructure.DataObjects.Events1d import Events1dBuilder
from DataStructure.DataObjects.Events2d import Events2dBuilder


class Filters:
    def __init__(self):
        pass

    @staticmethod
    def filter(obj, filter_obj, name, label=None, category=None, description=None):
        index = filter_obj.df.index
        df = obj.df.loc[index, :]
        return Events1dBuilder.build(
            df=df, name=name, category=category, label=label, description=description)

    @staticmethod
    def filter_with_value(
            obj, filter_obj, filter_values, name,
            xname=None, yname=None,
            label=None, xlabel=None, ylabel=None, category=None, description=None):
        df_filter = filter_obj.df == filter_values[0]
        for filter_val in filter_values[1:]:
            df_filter *= filter_obj.df == filter_val
        df_filter = df_filter[df_filter].dropna()
        idx_filter = df_filter.index
        df = obj.df.loc[idx_filter, :]
        df.dropna(inplace=True)

        n_dim = len(obj.df.columns)
        if n_dim == 1:
            return Events1dBuilder.build(
                df=df, name=name, category=category, label=label, description=description)
        elif n_dim == 2:
            if xname is None:
                xname = obj.df.columns[0]
            if yname is None:
                yname = obj.df.columns[1]

            return Events2dBuilder.build_from_df(
                df=df, name=name, xname=xname, yname=yname,
                category=category, label=label, xlabel=xlabel, ylabel=ylabel,
                description=description)
        else:
            raise ValueError(str(n_dim)+' is not a correct dimension')
