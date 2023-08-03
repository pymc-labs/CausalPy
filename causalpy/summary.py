"Summary objects."

from dataclasses import dataclass
from typing import Optional


def html_header(columns: list[str], colspan: int = 1) -> str:
    """
    Returns HTML code for table header.

    Parameters
    ----------
    columns :   list[str].
        List containing column names.
    colspan :   int.
        Column span.
    """
    string = (
        f'<th colspan={colspan} style="text-align: center"> '
        + (f'</th> <th style="text-align: center" colspan={colspan}>'.join(columns))
        + "</th>"
    )
    return '<tr style="text-align: center">' + string + "</tr>"


def html_rows(index: str, values: list, colspan: int = 1) -> str:
    """
    Returns HTML code for table rows.

    Parameters
    ----------
    items :   dict.
        The keys of items is the index-set of the rows to be added. The values are
        lists containing the values to be added.
    colspan :   int.
        Column span.
    """
    string = ""
    string += f'<th style="text-align: left"> {index} </th>'

    if not isinstance(values, list):
        values = [values]

    values = map(str, values)
    string += (
        f'<td colspan={colspan} style="text-align: center">'
        + ('</td> <td style="text-align: center">'.join(values))
        + "</td>"
    )

    string = f"<tr> {string} </tr>"

    return string


def str_header(columns: list[str], length: int) -> str:
    """
    Returns table header string.

    Parameters
    ----------
    columns :   list[str].
        List containing column names.
    length :   int.
        Length of box containing header.
    """
    # If first column is empty, it's box should not be displayed
    first_col_empty = columns[0] == ""

    if first_col_empty:
        columns = columns[1:]

    n_cols = len(columns)

    top = f"{length * '═'}╦"
    bot = f"{length * '═'}╩"

    spaces = first_col_empty * (length + 1) * " "

    return f"""\
    {spaces}╔{(n_cols - 1) * top}{length * "═"}╗
    {spaces}║{"║".join(map(lambda x: x.center(length), columns))}║
    ┏{length * '━'}╚{(n_cols - 1) * bot}{length * "═"}╝\
    """


def str_row(index: str, values: str, length: int, row_type: str = "inner") -> str:
    """
    Returns string table row.

    Parameters
    ----------
    index : str.
        Index of the row.
    values :    int.
        Values of the row.
    length :    int.
        Length of boxes.
    row_type :  str.
        One of "inner", "first", "last".
    """
    n_cols = len(values)
    values = map(str, values)

    s = ""

    if row_type == "first":
        s += f"""    ┏{length * '━'}┱"""
        s += f"{(n_cols - 1) * (length * '─' + '┬')}"
        s += f"{(length * '─' + '┐')}"
        s += "\n"

    s += f"""\
    ┃{index.center(length)}┃{"│".join(map(lambda x: x.center(length), values))}│
    """
    if row_type == "last":
        s += f"┗{length * '━'}┹{(n_cols - 1) * (length * '─' + '┴')}{length * '─'}┘"
    else:
        s += f"┣{length * '━'}╋{(n_cols - 1) * (length * '─' + '┼')}{length * '─'}┤"

    return s


def str_title(title: str):
    """
    Returns title in a box.

    Parameters
    ----------
    title : str.
        String to return in a box.
    """
    length = len(title)
    return f"""\
    ╔{(length + 2) * '═'}╗
    ║ {title} ║
    ╚{(length + 2) * '═'}╝
    """


@dataclass
class Record:
    """
    Class representing either a header or a row of a table.

    Parameters
    ----------
    record_type : str.
        Either 'header', 'title' or 'row'.
    values : list.
        List of values in record.
    colspan : int.
        Column span.
    """

    record_type: str
    values: list
    colspan: int
    index: Optional[str] = None

    def to_html(self):
        "Returns HTML code for record."
        if self.record_type in ["header", "title"]:
            return html_header(self.values, self.colspan)
        else:
            return html_rows(self.index, self.values, self.colspan)


class Summary:
    """
    Base summary class.
    """

    def __init__(self):
        self.records = []

    def add_header(self, col_names, colspan) -> None:
        "Adds a header."
        self.records.append(Record("header", col_names, colspan))

    def add_row(self, index: str, values: list, colspan: int) -> None:
        "Adds a row."
        self.records.append(Record("row", values, colspan, index=index))

    def add_title(self, title) -> None:
        "Adds title."
        self.records.append(Record("title", title, 1))

    def get_longest_length(self) -> int:
        """
        Returns the lenght of the longest item currently in the table.
        """
        non_titles = [r for r in self.records if not r.record_type == "title"]

        def length_of_items(L):
            return [len(str(x)) for x in L]

        vals = [length_of_items(r.values) for r in non_titles]
        longest_value_length = max([max(x) for x in vals])
        indx = [len(str(x.index)) for x in self.records]
        longest_index_length = max(indx)
        return max(longest_index_length, longest_value_length)

    def to_string(self) -> str:
        """
        Return string summary table in string form.
        """
        type_of_next_row = "first"
        length = self.get_longest_length()
        s = ""

        for i, x in enumerate(self.records):
            if i == len(self.records) - 1:
                type_of_next_row = "last"
            elif self.records[i + 1].record_type == "title":
                type_of_next_row = "last"

            if x.record_type == "header":
                s += str_header(x.values, length)
                type_of_next_row = "inner"
            elif x.record_type == "title":
                s += str_title(x.values[0])
                type_of_next_row = "first"
            else:
                s += str_row(x.index, x.values, length, type_of_next_row)
                type_of_next_row = "inner"
            s += "\n"
        return s

    def to_html(self) -> str:
        """
        Returns HTML code for summary table.
        """
        s = ""

        for x in self.records:
            s += x.to_html()

        return "<table>" + s + "</table>"

    def _repr_html_(self) -> str:
        return self.to_html()

    def __repr__(self) -> str:
        return self.to_string()
