# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
from typing import Any, Dict, List, Tuple

import pytest

from camel.functions.unstructured_io_fuctions import UnstructuredModules


# Create a fixture to initialize the UnstructuredModules instance
@pytest.fixture
def unstructured_instance() -> UnstructuredModules:
    return UnstructuredModules()


# Test the ensure_unstructured_version method
def test_ensure_unstructured_version(
        unstructured_instance: UnstructuredModules):
    # Test with a valid version
    unstructured_instance.ensure_unstructured_version("0.10.30")

    # Test with an invalid version (should raise a ValueError)
    with pytest.raises(ValueError):
        unstructured_instance.ensure_unstructured_version("1.0.0")


# Test the parse_file_or_url method
def test_parse_file_or_url(unstructured_instance: UnstructuredModules):
    # You can mock the required dependencies and test different scenarios here

    # Test parsing a valid URL (mock the necessary dependencies)
    result = unstructured_instance.parse_file_or_url(
        "https://www.cnn.com/2023/01/30/sport/empire-state-building-green-"
        "philadelphia-eagles-spt-intl/index.html")
    assert isinstance(result, list)

    # Test parsing a non-existent file (should raise FileNotFoundError)
    with pytest.raises(FileNotFoundError):
        unstructured_instance.parse_file_or_url("nonexistent_file.txt")


# Test the clean_text_data method
def test_clean_text_data(unstructured_instance: UnstructuredModules):
    # Test with a valid cleaning option
    test_options: List[Tuple[str,
                             Dict[str,
                                  Any]]] = [("clean_extra_whitespace", {})]
    cleaned_text = unstructured_instance.clean_text_data(
        text="  Hello  World  ", clean_options=test_options)
    assert cleaned_text == "Hello World"

    # Test with default cleaning options (no options provided)
    default_cleaned_text = unstructured_instance.clean_text_data(
        text="\x88  Hello  World  ")
    assert default_cleaned_text == "Hello World"

    # Test with an invalid cleaning option (should raise ValueError)
    test_options = [("invalid_cleaning_option", {})]
    with pytest.raises(ValueError):
        unstructured_instance.clean_text_data(text="Test Text",
                                              clean_options=test_options)


# Test the extract_data_from_text method
def test_extract_data_from_text(unstructured_instance: UnstructuredModules):
    # Test extracting an email address
    test_email_text = "Contact me at example@email.com."
    extracted_email = unstructured_instance.extract_data_from_text(
        text=test_email_text, extract_type="extract_email_address")
    assert extracted_email == ["example@email.com"]

    # Test with an invalid extract option (should raise ValueError)
    test_extract_type = "invalid_extracting_option"
    with pytest.raises(ValueError):
        unstructured_instance.extract_data_from_text(
            text=test_email_text,
            extract_type=test_extract_type)  # type: ignore


# Test the stage_elements method
def test_stage_elements_for_csv(unstructured_instance: UnstructuredModules):
    # Test staging for baseplate
    test_url = (
        "https://www.cnn.com/2023/01/30/sport/empire-state-building-green-"
        "philadelphia-eagles-spt-intl/index.html")
    test_elements = unstructured_instance.parse_file_or_url(test_url)
    staged_element: Any = unstructured_instance.stage_elements(
        elements=test_elements, stage_type="stage_for_baseplate")
    assert staged_element['rows'][0] == {
        'data': {
            'type': 'UncategorizedText',
            'element_id': 'e78902d05b0cb1e4c38fc7a79db450d5',
            'text': 'CNN\n        \xa0—'
        },
        'metadata': {
            'filetype':
            'text/html',
            'languages': ['eng'],
            'page_number':
            1,
            'url':
            'https://www.cnn.com/2023/01/30/sport/'
            'empire-state-building-green-philadelphia-eagles-spt-'
            'intl/index.html',
            'emphasized_text_contents': ['CNN'],
            'emphasized_text_tags': ['span']
        }
    }

    # Test with an invalid stage option (should raise ValueError)
    test_stage_type = "invalid_stageing_option"
    with pytest.raises(ValueError):
        unstructured_instance.stage_elements(
            elements=test_elements, stage_type=test_stage_type)  # type: ignore


# Test the chunk_elements method
def test_chunk_elements(unstructured_instance: UnstructuredModules):
    # Test chunking content from a url
    test_url = (
        "https://www.cnn.com/2023/01/30/sport/empire-state-building-green-"
        "philadelphia-eagles-spt-intl/index.html")
    test_elements = unstructured_instance.parse_file_or_url(test_url)
    chunked_sections = unstructured_instance.chunk_elements(
        elements=test_elements, chunk_type="chunk_by_title")

    assert len(chunked_sections) == 7  # Check the number of chunks
    # Test with an invalid chunk option (should raise ValueError)
    test_chunk_type = "chunk_by_invalid_option"
    with pytest.raises(ValueError):
        unstructured_instance.chunk_elements(elements=test_elements,
                                             chunk_type=test_chunk_type)
