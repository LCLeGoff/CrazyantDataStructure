from DataStructure.Builders.ExperimentGroupBuilder import ExperimentGroupBuilder
from ExperimentGroups import ExperimentGroups
from Scripts.root import root


class AnalyseClassDecorator:
    def __init__(self, group, exp: ExperimentGroups = None):
        if exp is None:
            self.exp = ExperimentGroupBuilder(root).build(group)
        else:
            self.exp = exp
        pass

    def compute_hist(self, result_name, bins, hist_name=None,
                     hist_description=None, hist_label=None, redo=False, redo_hist=False):
        if redo is True or redo_hist is True:
            self.exp.load(result_name)
            hist_name = self.exp.hist1d(name_to_hist=result_name, result_name=hist_name,
                                        bins=bins, label=hist_label, description=hist_description)
            self.exp.write(hist_name)

        else:
            if hist_name is None:
                hist_name = result_name+'_hist'
            self.exp.load(hist_name)

        return hist_name
