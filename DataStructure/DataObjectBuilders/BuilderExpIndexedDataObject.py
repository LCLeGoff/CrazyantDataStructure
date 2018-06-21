class BuilderExpIndexedDataObject:
    def __init__(self, df):
        if df.index.names != ['id_exp']:
            raise IndexError('Index names are not id_exp')
        else:
            self.df = df

    def get_row_of_id_exp(self, id_exp):
        return self.df.loc[id_exp, :]

    def operation_on_id_exp(self, id_exp, fct):
        self.df.loc[id_exp] = fct(self.df.loc[id_exp])
