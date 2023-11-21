import re
import warnings
from typing import Any
from typing import List
from typing import Optional
from typing import Union

import gspread
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


DEFAULT_EMPTY_VALUE = ""


class ResultNotFound(Exception):
    pass


class MultipleResultsFound(Exception):
    pass


class SpreadsheetNotFound(Exception):
    pass


class WorksheetNotFound(Exception):
    pass


class InvalidConfiguration(Exception):
    pass


class FieldNameMismatch(Exception):
    pass


def find_ranges(sequence: List[int]) -> List[List[int]]:
    if not sequence:
        return []

    sequence.sort()
    ranges = []
    start = sequence[0]

    for i in range(len(sequence) - 1):
        if sequence[i] + 1 < sequence[i + 1]:
            ranges.append(list(range(start, sequence[i] + 1)))
            start = sequence[i + 1]

    ranges.append(list(range(start, sequence[-1] + 1)))

    return ranges


def split_list(original, lengths):
    result = []
    start = 0

    for length in lengths:
        end = start + length
        result.append(original[start:end])
        start = end

    return result


class QueryManager:
    def __init__(self, model: "GSheetModel") -> None:
        self.model = model
        self.data_is_a_range = False
        self.headers_col = {}

    @property
    def gc(self) -> gspread.client.Client:
        if not hasattr(self, "_gc"):
            try:
                self._gc = gspread.service_account_from_dict(self.model.Meta.configuration)
            except Exception:
                raise InvalidConfiguration("Invalid configuration.")
        return self._gc

    @property
    def spreadsheet(self):
        if not hasattr(self, "_spreadsheet"):
            try:
                if hasattr(self.model.Meta, "spreadsheet_id"):
                    self._spreadsheet = self.gc.open_by_key(self.model.Meta.spreadsheet_id)
                elif hasattr(self.model.Meta, "spreadsheet_name"):
                    self._spreadsheet = self.gc.open(self.model.Meta.spreadsheet_name)
                else:
                    # Assume url
                    self._spreadsheet = self.gc.open_by_url(self.model.Meta.spreadsheet_url)
            except gspread.SpreadsheetNotFound as e:
                raise SpreadsheetNotFound(e)
        return self._spreadsheet

    @property
    def worksheet(self):
        if not hasattr(self, "_worksheet"):
            if (
                not hasattr(self.model, "Meta")
                or self.model.Meta is None
                or not hasattr(self.model.Meta, "worksheet_name")
                or self.model.Meta.worksheet_name is None
            ):
                self._worksheet = self.spreadsheet.get_worksheet(0)  # Get the first worksheet
            else:
                try:
                    self._worksheet = self.spreadsheet.worksheet(self.model.Meta.worksheet_name)
                except gspread.WorksheetNotFound:
                    raise WorksheetNotFound("Worksheet not found.")

        if not self._worksheet:
            raise WorksheetNotFound("Worksheet not found or specified.")

        return self._worksheet

    @property
    def headers(self, force: Optional[bool] = False) -> List:
        if not hasattr(self, "_headers") or force:
            if not self.worksheet.row_values(1):
                # add header row to worksheet if it's empty
                self.worksheet.append_row(self.model.fields(exclude={"id"}))

            self._headers = self.worksheet.row_values(1)
            if not set(self.model.fields(exclude={"id"})).issubset(set(self._headers)):
                raise FieldNameMismatch("Field names in model and worksheet do not match.")

        return self._headers

    @property
    def header_to_col(self):
        if not hasattr(self, "_header_to_col"):
            self._header_to_col = {}
            for key in self.model.fields(exclude={"id"}):
                cell = self.worksheet.find(key)
                self._header_to_col[key] = cell.col
        return self._header_to_col

    @property
    def col_to_header(self):
        if not hasattr(self, "_col_to_header"):
            self._col_to_header = {value: key for key, value in self.header_to_col.items()}
        return self._col_to_header

    @property
    def col_ranges(self):
        """Retrieves ranges of columns so we can optimize the number of updating calls"""
        if not hasattr(self, "_col_ranges"):
            self._col_ranges = find_ranges(list(self.header_to_col.values()))
        return self._col_ranges

    def get(self, **kwargs):
        """Fetch just one row from the spreadsheet"""
        found_obj = None
        for obj in self.query_generator(**kwargs):
            if found_obj is not None:
                # we already found one row, so another one is not allowed
                raise MultipleResultsFound("Multiple objects returned.")
            found_obj = obj

        # check if any results
        if found_obj is None:
            raise ResultNotFound("No results found.")

        return found_obj

    def query(self, **kwargs):
        """Fetch all rows from the spreadsheet"""
        _records = list(self.query_generator(**kwargs))
        if not _records:
            raise ResultNotFound("No results found.")
        return _records

    def query_generator(self, **kwargs):
        # TO-DO: Improve this method's performance
        for obj in self.all():
            # obj_dict = json.loads(obj.model_dump_json())
            obj_dict = obj.model_dump()
            if all(obj_dict.get(k) == v for k, v in kwargs.items()):
                yield obj

    def count(self):
        return sum(1 for _ in self.all())

    def get_all_records(self):
        # Wrapper around get_all_records() because it has problems when the doc doesn't have any rows other than the headers
        try:
            return self.worksheet.get_all_records(default_blank=None)
        except IndexError:
            r = self.worksheet.get_all_values()
            if len(r) < 2:
                return []
            raise

    def all(self):
        for idx, raw_row in enumerate(self.get_all_records()):
            obj = self.model.parse_row(raw_row)
            obj._disable_change_tracking()
            obj.id = idx + 2  # due to items starting at 2
            obj._enable_change_tracking()
            yield obj


class GSheetModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True, json_encoders={None: lambda _: DEFAULT_EMPTY_VALUE})

    id: Optional[int] = Field(None, gt=0)

    # private fields
    _changed_fields: set = set()
    _objects: Optional[Any] = None

    class Meta:
        spreadsheet_id = None
        spreadsheet_name = None
        spreadsheet_url = None
        worksheet_name = None
        configuration = {}  # JSON config as dictionary

    @property
    def objects(self):
        return self._objects

    @objects.setter
    def objects(self, value):
        self._objects = value

    @classmethod
    def fields(cls, exclude: Optional[Union[List, set]] = None):
        """Returns a list of field names"""
        if exclude is None:
            exclude = ()

        _fields = list(cls.model_fields)
        for key in _fields:
            if key in exclude:
                _fields.remove(key)
        return _fields

    @classmethod
    def parse_row(cls, row):
        """Translates from a dict to an instance of the model"""
        for field in row:
            if not row[field]:
                if field in cls.__annotations__:
                    row[field] = cls.__annotations__[field].default if hasattr(cls.__annotations__[field], "default") else None
                else:
                    row[field] = None
        data = {k: v for k, v in row.items() if k in cls.fields(exclude={"id"})}
        return cls.model_validate(data)  # only fields defined in the model

    def _get_row_id(self, descriptor):
        # descriptor is like DB!A4:C4 or 'Sheet1!A4'
        rx = re.compile(r"[^!]+![A-Z]+(\d+):?")
        m = rx.search(descriptor)
        if m:
            return int(m.group(1))
        raise ValueError("Not a valid range descriptor")

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.objects = QueryManager(model=cls)

    def __setattr__(self, key, value):
        # If the key is not one of the private fields, check for changes
        if not key.startswith("_") and getattr(self, "_change_tracking_enabled", True):
            # Compare with the current value
            current_value = getattr(self, key, None)
            if current_value != value:
                self._changed_fields.add(key)

        # Call the super().__setattr__ method to actually set the value
        super().__setattr__(key, value)

    def _disable_change_tracking(self):
        super().__setattr__("_change_tracking_enabled", False)

    def _enable_change_tracking(self):
        super().__setattr__("_change_tracking_enabled", True)

    def has_changed(self):
        # Check if any fields have been changed
        return bool(self._changed_fields)

    def save(self):
        self.__class__(**self.model_dump())  # re-create instance to trigger validation
        # trigger setting first row if not there yet
        assert bool(self.objects.headers)

        # turn the object into a row in the order of the spreadsheet
        dumped = self.model_dump(exclude={"id"}, mode="json")

        # save data
        if self.id is None:
            # this is a new object, must append_row using all fields of headers although non-model fields must be empty
            full_values = {k: None for k in self.objects.worksheet.row_values(1)}
            full_values.update(dumped)
            up = self.objects.worksheet.append_row([full_values[field] for field in self.objects.headers])
            self.id = self._get_row_id(up["updates"]["updatedRange"])
        else:
            # this is an existing object
            # update just our fields
            values = [dumped[field] for field in self.objects.headers if field in self.fields()]
            splitted_values = split_list(values, [len(r) for r in self.objects.col_ranges])

            data = []
            for col_range, range_values in zip(self.objects.col_ranges, splitted_values):
                range = "{start}:{end}".format(
                    start=gspread.utils.rowcol_to_a1(self.id, col_range[0]), end=gspread.utils.rowcol_to_a1(self.id, col_range[-1])
                )
                data.append({"range": range, "values": [range_values]})
            self.objects.worksheet.batch_update(data)

    def delete(self):
        if self.id is not None:
            self.objects.worksheet.delete_row(self.id)
            self.id = None
        else:
            warnings.warn("Object is not persisted yet so nothing to delete.")
