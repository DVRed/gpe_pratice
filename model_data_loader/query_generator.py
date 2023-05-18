from model_data_loader_c.formula_parser import DataSource, SumIfFormula, Condition
from model_data_loader_c.utils import indent_with_tabs
from datetime import datetime


def generate_cross_tab(data_source: DataSource):
    """
    Генерирует CROSSTAB sql-запрос, множество значений колонки cross_tab_property развертываются в колонки,
    значения в этих колонках берутся из колонки value исходной таблицы
    Исходная таблица берется из запроса data_source.source_query
    Идентификатор строки - date и колонки, отличные от cross_tab_property и value
    """

    cross_tab_property = data_source.get_most_relevant_property().property_name
    cross_tab_column_list = data_source.get_most_relevant_property().value_set

    category_query = 'select ' + cross_tab_property + ' from (values'
    select_from_source_query = ''

    extra_properties = list(filter(
        lambda property_name0:
        property_name0 != 'value' and
        property_name0 != cross_tab_property and
        property_name0 != 'date',
        data_source.required_properties_dict.keys()  # data_source.required_properties
    ))

    # data_source.required_properties

    if len(extra_properties) != 0:
        select_from_source_query = 'select CONCAT(date, '
        for prop in extra_properties:
            select_from_source_query += prop
            if prop != extra_properties[-1]:
                select_from_source_query += ', '
        select_from_source_query += ') as id, '
    else:
        select_from_source_query = 'select date as id, '

    select_from_source_query += 'date, '

    cross_tab_column_select = '"id" varchar,\n"date" timestamp without time zone,\n'

    for property_name in data_source.required_properties_dict.keys():  # data_source.required_properties:
        print(property_name)
        if property_name != 'value' and property_name != cross_tab_property:  # 'point_name':
            if property_name == 'date':
                print('AAAAAAAAAA')
                continue
                # cross_tab_column_select += '"date" timestamp without time zone'
            else:
                cross_tab_column_select += '"' + property_name + '" varchar'
            cross_tab_column_select += ',\n'
        if property_name != 'date' and property_name != 'value' and property_name != cross_tab_property: # 'point_name':
            select_from_source_query += property_name + ', '

    for i, property_value in enumerate(cross_tab_column_list):  # data_source.required_point_names):
        cross_tab_column_select += '"' + property_value + '" numeric'
        category_query += '(\'\'' + property_value + '\'\')'
        if i != len(cross_tab_column_list) - 1:  # data_source.required_point_names) - 1:
            cross_tab_column_select += ',\n'
            category_query += ','

    select_from_source_query += \
        cross_tab_property + ', value\n' \
        'from (\n' + \
        indent_with_tabs(data_source.source_query.replace("'", "''"), 1) + '\n' + \
        ') m order by id'

    category_query += ') b(' + cross_tab_property + ')'  # point_name)'

    return 'select * from crosstab(\n' + \
        indent_with_tabs("'" + select_from_source_query + "'", 1) + ',\n' + \
        indent_with_tabs("'" + category_query + "'", 1) + ') \nas ct(\n' + \
        indent_with_tabs(cross_tab_column_select, 1) + '\n)'


def generate_query(data_sources: list[DataSource],
                   sum_if_formulas: list[SumIfFormula],
                   column_names: list[str],
                   begin_date: datetime,
                   end_date: datetime
                   ) -> str:
    data_source_query = 'with '
    for data_source in data_sources:
        source_query = ''
        # date,
        if len(data_source.required_properties_dict.keys()) > 1:
            print('generating crosstab query for ' + data_source.identifier)
            source_query = generate_cross_tab(data_source)  # TODO
        else:
            source_query = data_source.source_query
        data_source_query += '\n' + data_source.identifier + ' as (\n' +\
            indent_with_tabs(source_query, 1) + '\n),'

    date_format = '%Y-%m-%d %H:%M:%S'
    begin_date_formatted = begin_date.strftime(date_format)
    end_date_formatted = end_date.strftime(date_format)
    # TODO handle if end_date < begin_date

    data_source_query += '\ndates as (' \
                         'select * from generate_series(' \
                         f'\'{begin_date_formatted}\'::timestamp,' \
                         f'\'{end_date_formatted}\'::timestamp,' \
                         ' \'1 day\'::interval) date)\n'

    select = 'select \n\tdates.date as "Date",\n'
    joins = ''
    t = 0
    j = 0

    for sum_if_formula in sum_if_formulas:
        cross_tab_property = sum_if_formula.data_source.get_most_relevant_property().property_name

        inner_alias = 't' + str(t)
        t += 1
        join_alias = 'j' + str(j)
        j += 1

        cross_tab_property_condition = None
        cross_tab_property_filtered_list: list[Condition] = list(filter(
            lambda condition0: condition0.argument.property_name == cross_tab_property,
            sum_if_formula.conditions)
        )
        if len(cross_tab_property_filtered_list) > 0:
            cross_tab_property_condition = cross_tab_property_filtered_list[0]

        other_conditions: list[Condition] = list(filter(
            lambda condition0:
                condition0.argument.property_name != cross_tab_property and
                condition0.argument.property_name != 'date',
            sum_if_formula.conditions
        ))

        joins += 'left join (\n\tselect '

        if cross_tab_property_condition is None:
            select_value = 'value'
        else:
            select_value = cross_tab_property_condition.value

        joins += '"' + select_value + '", date\n'
        joins += '\tfrom ' + sum_if_formula.sum_argument.identifier + ' ' + inner_alias

        if len(other_conditions) != 0:
            joins += '\n\twhere '
            for condition in other_conditions:
                joins += inner_alias + '.' + condition.argument.property_name + '=\'' + condition.value + '\''
                if condition != other_conditions[-1]:
                    joins += ' and '

        joins += '\n) ' + join_alias + ' on ' + join_alias + '.date=dates.date\n'

        select += '\t' + join_alias + '."' + select_value + '"' + sum_if_formula.multipliers +\
            ' as "' + column_names[sum_if_formula.column_index] + '"'

        if sum_if_formula != sum_if_formulas[-1]:
            select += ',\n'

    return data_source_query + '\n' + select + '\n from dates\n' + joins
