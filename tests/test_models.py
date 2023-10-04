from unittest.mock import Mock
from unittest.mock import patch

import gspread
import pytest

from gspread_orm.models import GSheetModel
from gspread_orm.models import InvalidConfiguration
from gspread_orm.models import MultipleResultsFound
from gspread_orm.models import QueryManager
from gspread_orm.models import ResultNotFound
from gspread_orm.models import SpreadsheetNotFound
from gspread_orm.models import WorksheetNotFound


TEST_DATA = [["name"], ["foo"], ["foo"], ["bar"]]  # this is the headers row  # id=2  # id=3  # id=4


def mock_fn_get_all_records():
    """Mocked function to simulate gspread.worksheet.get_all_records method"""
    records = []
    for idx in range(1, len(TEST_DATA)):
        records.append({"name": TEST_DATA[idx][0]})
    return records


def mock_fn_row_values(idx):
    """Mocked function to simulate gspread.worksheet.row_count method"""
    return TEST_DATA[idx - 1]


@pytest.fixture()
def mock_gspread_service_account():
    """mock all gspread calls so they work as required by the tests"""
    with patch("gspread.service_account_from_dict", return_value=Mock()) as mock_gspread_action_service:
        mock_worksheet = Mock()
        mock_worksheet.row_count = len(TEST_DATA)
        mock_worksheet.row_values = Mock()
        mock_worksheet.row_values.side_effect = mock_fn_row_values
        mock_worksheet.get_all_values = Mock()
        mock_worksheet.get_all_values.return_value = TEST_DATA
        mock_worksheet.get_all_records = Mock()
        mock_worksheet.get_all_records.side_effect = mock_fn_get_all_records
        mock_spreadsheet = Mock()
        mock_spreadsheet.worksheet.return_value = mock_worksheet
        mock_service = Mock()
        mock_service.open_by_key.return_value = mock_spreadsheet
        mock_gspread_action_service.return_value = mock_service
        yield mock_gspread_action_service


@pytest.fixture()
def mock_cls():
    """Provide a fresh new instance of a subclass on each test function"""

    class DummyModel(GSheetModel):
        name: str

        class Meta:
            configuration = {}
            spreadsheet_id = "spreadsheet_id"
            worksheet_name = "worksheet_name"

    yield DummyModel


# Assuming mock_gspread_service_account is defined in conftest.py
def test_init(mock_gspread_service_account):
    """Tests initialization of GSheetModel"""
    test_instance = GSheetModel()
    assert test_instance.objects is None  # objects is initializated exclusively in subclasses
    assert list(test_instance.fields()) == ["id"]
    assert test_instance.id is None


def test_subclass(mock_gspread_service_account, mock_cls):
    """Tests initialization of a subclass"""
    test_instance = mock_cls(name="test")  # name is required
    assert isinstance(test_instance.objects, QueryManager)  # this time .objects is QueryManager
    assert test_instance.id is None  # still .id is None


def test_count(mock_gspread_service_account, mock_cls):
    assert mock_cls.objects.count() == len(TEST_DATA) - 1  # -1 because the first row is headers


def test_all(mock_gspread_service_account, mock_cls):
    assert len(list(mock_cls.objects.all())) == len(TEST_DATA) - 1  # -1 because the first row is headers


def test_error_invalid_configuration(mock_gspread_service_account, mock_cls):
    mock_gspread_service_account.side_effect = Exception("Some error")

    with pytest.raises(InvalidConfiguration):
        mock_cls.objects.get(id=2)


def test_error_spreadsheet_not_found(mock_gspread_service_account, mock_cls):
    mock_gspread_service_account.return_value.open_by_key.side_effect = gspread.SpreadsheetNotFound("cuek")

    with pytest.raises(SpreadsheetNotFound):
        mock_cls.objects.get(id=2)


def test_error_worksheet_not_found(mock_gspread_service_account, mock_cls):
    mock_gspread_service_account.return_value.open_by_key.return_value.worksheet.side_effect = gspread.WorksheetNotFound("Some error")

    with pytest.raises(WorksheetNotFound):
        mock_cls.objects.get(id=2)


def test_error_result_not_found(mock_gspread_service_account, mock_cls):
    mock_gspread_service_account.return_value.open_by_key.return_value.worksheet.return_value.row_values.side_effect = lambda x: []

    with pytest.raises(ResultNotFound):
        mock_cls.objects.get(id=1)


def test_error_multiple_results_found(mock_gspread_service_account, mock_cls):
    with pytest.raises(MultipleResultsFound):
        mock_cls.objects.get(name="foo")


def test_get_by_id(mock_gspread_service_account, mock_cls):
    o = mock_cls.objects.get(id=3)
    assert o.id == 3
    assert o.name == "foo"


def test_get_ok(mock_gspread_service_account, mock_cls):
    o = mock_cls.objects.get(name="bar")
    assert o.id == 4


def test_query_ok(mock_gspread_service_account, mock_cls):
    o = mock_cls.objects.query(name="foo")
    assert len(o) == 2


def test_error_query_result_not_found(mock_gspread_service_account, mock_cls):
    with pytest.raises(ResultNotFound):
        mock_cls.objects.query(name="this name is not stored")
