from AnalyseClasses.AnalyseClassDecorator import AnalyseClassDecorator


class UOModelAnalysis(AnalyseClassDecorator):
    def __init__(self, group, exp=None):
        AnalyseClassDecorator.__init__(self, group, exp)
        self.category = 'Models'

    def compute_model_attachment_intervals(self, model_name, parameter_list=None):

        name = model_name+'Attachments'
        self.exp.load(name)

        if parameter_list is None:
            parameter_list = list(self.exp.get_data_object(name).get_column_names())

        for para in parameter_list:
            c, p, so, si = para.split(', ')
            c = c[1:]
            si = si[:-1]
            result_name = '%s_c%s_p%s_so%s_si%s_attachment_intervals' % (name, c, p, so, si)

            label = 'attachment intervals of %s' % model_name
            description = 'attachment intervals of %s of parameters (c, p, sigma_orient, sigma_info)=%s' % (name, para)

            tab = self.exp.get_df(name)[para].values.ravel()
            temp_name = 'temp'
            self.exp.add_new1d_from_array(array=tab, name=temp_name, object_type='CharacteristicTimeSeries1d',
                                          replace=True)

            df_interval = self.exp.compute_time_intervals(name_to_intervals=name, category=self.category,
                                                          result_name=result_name, label=label, description=description)

            self.exp.add_new_dataset_from_df(df=df_interval, name=result_name, category=self.category,
                                             label=label, description=description)

            self.exp.write(result_name)
            self.exp.remove_object(result_name)

    # def compute_mm1s_food_direction_error_around_outside_attachments(self):
    #         attachment_name = 'outside_ant_carrying_intervals'
    #         variable_name = 'mm1s_food_direction_error'
    #
    #         result_name = variable_name + '_around_outside_attachments'
    #
    #         result_label = 'Food direction error around outside attachments'
    #         result_description = 'Food direction error smoothed with a moving mean of window ' + str(mm) + ' s for times' \
    #                              ' before and after an ant coming from outside ant attached to the food'
    #
    #         self.__gather_variable_around_attachments(variable_name, attachment_name, result_name, result_label,
    #                                                   result_description)