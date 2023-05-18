import re

QUERIES_PATH = 'queries/'  # TODO remove as redundant


class Argument:
    """
    Класс, представляющий аргумент в условии.
    Напр. Table_External_Data[some_property]

    Attributes:
        identifier (str): имя внешней таблицы
        property_name (str): свойство (столбец)
    """
    def __init__(self, identifier: str, property_name: str):
        self.identifier = identifier
        self.property_name = property_name

    def __eq__(self, obj):
        return isinstance(obj, Argument) and obj.identifier == self.identifier


class Condition:
    """
    Класс представляющий условие в excel формуле: Criteria_range; Criteria.
    Напр. Table_External_Data[curve_name], "exit"
    Attributes:
        argument (Argument): Criteria_range
        value (str): Criteria, только строки, значения хранятся без кавычек
    """
    def __init__(self, argument: Argument, value: str):
        self.argument = argument
        self.value = value


class Property:
    """
    Столбец во внешней таблице DataSource

    Attributes:
        property_name (str): имя свойства
        value_set (set[str]): множество значений, которые
        number_of_usages (int): кол-во обращений в формулах к данному свойству
    """
    def __init__(self, property_name):
        self.property_name = property_name
        self.value_set = set([])
        self.number_of_usages = 0

    def add_value(self, str_value: str):
        self.number_of_usages += 1
        self.value_set.add(str_value)


class DataSource:
    """
    Внешняя таблица, на которую ссылается аргумент в формуле

    Attributes:
        identifier: имя внешней таблицы, которое упоминается в поле "Имя" закладки
                    "Область Использования" в свойствах подключения
        source_query: SQL запрос, формирующий внешнюю таблицу
    """
    # TODO identifier объявленный в поле "Имя" закладки "Область Использования" в свойствах подключения
    #  может отличаться от имени, которое упоминается в формуле
    def __init__(self, identifier: str, source_query: str, required_point_names=None, required_properties=None, required_properties_dict=None):
        # TODO remove as we have required_properties dict
        # if required_properties is None:
        #     required_properties = set([])  # TODO order is not guarantee
        # if required_point_names is None:
        #     required_point_names = set([])  # TODO order is not guarantee
        #
        if required_properties_dict is None:
            required_properties_dict: dict[str, Property] = {}
        self.identifier = identifier
        self.source_query = source_query
        # with open(QUERIES_PATH + identifier + '_query', 'r') as file:
        #     self.source_query = file.read()  # TODO Fluxys_data_query today() date
        # self.required_point_names = required_point_names
        self.required_properties_dict = required_properties_dict
        # self.required_properties = required_properties
        self.most_relevant_property = None

    def __eq__(self, obj):
        return isinstance(obj, DataSource) and obj.identifier == self.identifier

    def get_most_relevant_property(self) -> Property:
        """
        Метод для получения наиболее выгодного для cross_tab'a свойства
        наиболее выгодным будет то, к которому обращались чаще и у которого наибольшая вариативность значений свойства
        """
        if self.most_relevant_property is not None:
            return self.most_relevant_property
        most_relevant = None
        most_relevant_score = -1
        for property_obj in self.required_properties_dict.values():
            current_score = property_obj.number_of_usages * len(property_obj.value_set)
            if current_score > most_relevant_score:
                most_relevant_score = current_score
                most_relevant = property_obj
        self.most_relevant_property = most_relevant
        return most_relevant


class SumIfFormula:
    """
    Класс, представляющий SUMIFS(Sum_range; Criteria_range1; Criteria1; ...)*a*b*... excel формулу
    Attributes:
        sum_argument: Sum_range
        conditions: Criteria_range1, Criteria1, ...
        data_source: определяется идентификатором в sum_argument
        multipliers: *a*b*...
        column_index: индекс столбца, в котором лежит формула
    """
    def __init__(self, data_source: DataSource, sum_argument: Argument,
                 conditions: list[Condition], column_index: int, multipliers: str = ''):
        self.data_source = data_source
        self.sum_argument = sum_argument
        self.column_index = column_index
        self.conditions = conditions
        self.multipliers = multipliers

    def __eq__(self, obj):
        return isinstance(obj, SumIfFormula) and obj.column_index == self.column_index


class FormulaParser:

    data_sources: dict[str, DataSource] = {}
    sum_if_formulas: list[SumIfFormula] = []

    MULTIPLIERS_PATTERN = r'(?<=\))\*.*'
    FUNCTION_BODY_PATTERN = r'(?<=SUMIFS\().+(?=\))'
    ARGUMENT_SPLITTER_PATTERN = r'[a-zA-Z_\"0-9 \\-]+'

    def __init__(self, formulas: list[str], connections):
        self.parse(formulas, connections)

    def parse(self, formulas: list[str], connections):
        for i, formula in enumerate(formulas):

            # formula = formula.replace('Table_ExternalData_164', 'House_curves_data')
            # formula = formula.replace('Table_ExternalData_16', 'Fluxys_data')
            formula = re.sub(r'Daily!\$[A-Z]{1,2}\d+', 'date', formula)
            formula = formula.replace('[[#All],', '')
            formula = formula.replace(']]', ']')
            formula = formula.replace('[1]', '')
            formula = formula.replace('!', '')
            # formula = formula.replaceAll(r'Daily!\\$[A-Z]{1,2}\\d+', 'date')  # TODO

            if re.search(r'=SUMIFS\(.*\)', formula):

                ds = None
                sum_argument = None
                conditions = []
                multipliers = ''

                multipliers_result = re.findall(self.MULTIPLIERS_PATTERN, formula)
                if len(multipliers_result) != 0:
                    multipliers = multipliers_result[0]

                function_body_result = re.findall(self.FUNCTION_BODY_PATTERN, formula)
                function_body = ''
                if len(function_body_result) != 0:
                    function_body = function_body_result[0]

                arguments = function_body.split(',')

                prev_argument = Argument('', '')

                for argument in arguments:
                    argument_parts = re.findall(self.ARGUMENT_SPLITTER_PATTERN, argument)

                    # Первый аргумент SUMIF формулы - диапозон суммирования
                    if argument == arguments[0]:
                        identifier = argument_parts[0]
                        property_name = argument_parts[1]
                        if identifier in self.data_sources:
                            ds = self.data_sources[identifier]
                        else:
                            ds = DataSource(identifier, connections[identifier][:-1])
                            self.data_sources[identifier] = ds
                        # ds.required_properties.add(property_name)
                        sum_argument = Argument(identifier, property_name)
                        continue

                    # Если число частей аргумента 1, то это значение критерия с которым
                    # сравнивается свойство, которое было передано ранее и сохранено в prev_argument
                    if len(argument_parts) == 1:
                        value = argument_parts[0].replace('"', '')
                        data_source = self.data_sources[prev_argument.identifier]
                        # if prev_argument.property_name == 'point_name':
                        #     data_source.required_point_names.add(value)
                        conditions.append(Condition(prev_argument, value))
                        # data_source.required_properties.add(prev_argument.property_name)

                        property_dict = data_source.required_properties_dict
                        if prev_argument.property_name not in property_dict.keys():
                            my_property = Property(prev_argument.property_name)
                            #self.data_sources[prev_argument.identifier].required_properties_dict[prev_argument.property_name] = my_property
                            property_dict[prev_argument.property_name] = my_property
                        else:
                            # my_property = self.data_sources[prev_argument.identifier].required_properties_dict[prev_argument.property_name]
                            my_property = property_dict[prev_argument.property_name]
                        my_property.add_value(value)

                    # Если число частей аргумента 2, то это ссылка на свойство внешней таблицы
                    # Table_External_data[some_property]
                    if len(argument_parts) == 2:
                        identifier = argument_parts[0]
                        property_name = argument_parts[1]
                        prev_argument = Argument(identifier, property_name)

                sif = SumIfFormula(ds, sum_argument, conditions, i, multipliers)
                self.sum_if_formulas.append(sif)
            formulas[i] = formula