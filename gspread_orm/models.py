import re
import warnings
from typing import Any
from typing import Optional

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


def get_column_letter(index):
    letters = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


class QueryManager:
    def __init__(self, model):
        self.model = model

    @property
    def gc(self):
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
                self._spreadsheet = self.gc.open_by_key(self.model.Meta.spreadsheet_id)
            except gspread.SpreadsheetNotFound as e:
                raise SpreadsheetNotFound(e)
        return self._spreadsheet

    @property
    def worksheet(self):
        if not hasattr(self, "_worksheet"):
            if not hasattr(self.model.Meta, "worksheet_name") or self.model.Meta is None:
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
    def headers(self):
        if not hasattr(self, "_headers"):
            if self.count() == 0:
                # add header row to worksheet if it's empty
                self.worksheet.append_row(self.model.fields(exclude={"id"}))

            self._headers = self.worksheet.row_values(1)
            if set(self._headers) != set(self.model.fields(exclude={"id"})):
                raise FieldNameMismatch("Field names in model and worksheet do not match.")
        return self._headers

    @property
    def col_to_field_map(self):
        if not hasattr(self, "_col_to_field_map"):
            self._col_to_field_map = {get_column_letter(i + 1): value for i, value in enumerate(self.headers)}
        return self._col_to_field_map

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

    def all(self):
        for idx, raw_row in enumerate(self.worksheet.get_all_records()):
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
        worksheet_name = None
        configuration = {}  # JSON config as dictionary

    @property
    def objects(self):
        return self._objects

    @objects.setter
    def objects(self, value):
        self._objects = value

    @classmethod
    def fields(cls, exclude=None):
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
        return cls.model_validate(row)

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

        # turn the object into a row in the order of the spreadsheet
        dumped = self.model_dump(exclude={"id"}, mode="json")
        values = [dumped[field] for _, field in self.objects.col_to_field_map.items()]

        # save data
        if self.id is None:
            # this is a new object
            up = self.objects.worksheet.append_row(values)
            self.id = self._get_row_id(up["updates"]["updatedRange"])
        else:
            # this is an existing object
            cols = list(self.objects.col_to_field_map.keys())
            cell_range = f"{cols[0]}{self.id}:{cols[-1]}{self.id}"
            self.objects.worksheet.update(cell_range, [values])

    def delete(self):
        if self.id is not None:
            self.objects.worksheet.delete_row(self.id)
            self.id = None
        else:
            warnings.warn("Object is not persisted yet so nothing to delete.")
